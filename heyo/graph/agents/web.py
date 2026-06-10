"""Web navigation agent: Playwright (headless Chromium) tools.

Requires the `web` extra: uv sync --extra web && uv run playwright install chromium
If Playwright is missing the agent reports that instead of crashing the graph.
"""

from __future__ import annotations

from heyo.graph.agents.base import ToolKit, make_tool_agent
from heyo.llm.client import LLMClient

DESCRIPTION = "browse the web: open pages, read their content, click links, extract info"

WEB_PROMPT = """\
You are Heyo's web-navigation agent driving a headless browser.

RULES — follow strictly:
1. NEVER answer from memory. Your training data is outdated. Your first action is
   ALWAYS a tool call: search(query), or goto(url) if the user gave a URL.
2. Typical flow: search(query) -> goto(a result url) -> read_page() -> answer.
3. Never invent URLs. If a page fails or is empty, try the next search result.
4. Base your final answer ONLY on text returned by your tools, quoting the page.
   If the tools returned nothing useful, say so — do not fill gaps from memory.
"""


# Headless Chromium's default UA triggers bot detection on most search engines.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class Browser:
    """Lazy singleton wrapper around an async Playwright Chromium page."""

    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self.page = None

    async def ensure(self):
        if self.page is None:
            from playwright.async_api import async_playwright

            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=True)
            self.page = await self._browser.new_page(user_agent=USER_AGENT)
        return self.page

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        self._pw = self._browser = self.page = None


def make_web_toolkit(browser: Browser) -> ToolKit:
    from urllib.parse import quote_plus

    kit = ToolKit()

    @kit.add(
        "search",
        "Search the web and get a list of results (title -> url)",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    async def search(query: str) -> str:
        # Mojeek: independent index, no bot captchas/poisoning (DuckDuckGo serves
        # captchas to headless browsers, Bing serves decoy results), direct links.
        page = await browser.ensure()
        await page.goto(
            f"https://www.mojeek.com/search?q={quote_plus(query)}",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        items = await page.eval_on_selector_all(
            "a.title",
            "els => els.slice(0, 8).map(e => e.textContent.trim() + ' -> ' + e.href)",
        )
        return "\n".join(items) or "no results"

    @kit.add(
        "goto",
        "Navigate the browser to a URL",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    )
    async def goto(url: str) -> str:
        page = await browser.ensure()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return f"at {page.url} — title: {await page.title()}"

    @kit.add(
        "read_page",
        "Return the visible text of the current page (truncated)",
        {"type": "object", "properties": {}, "required": []},
    )
    async def read_page() -> str:
        page = await browser.ensure()
        text = await page.inner_text("body")
        return " ".join(text.split())[:8000]

    @kit.add(
        "click",
        "Click an element by CSS selector or by its visible text",
        {
            "type": "object",
            "properties": {"target": {"type": "string", "description": "CSS selector or link text"}},
            "required": ["target"],
        },
    )
    async def click(target: str) -> str:
        page = await browser.ensure()
        try:
            await page.click(target, timeout=5000)
        except Exception:
            await page.get_by_text(target).first.click(timeout=5000)
        await page.wait_for_load_state("domcontentloaded")
        return f"clicked; now at {page.url} — title: {await page.title()}"

    @kit.add(
        "links",
        "List links (text -> href) visible on the current page",
        {"type": "object", "properties": {}, "required": []},
    )
    async def links() -> str:
        page = await browser.ensure()
        items = await page.eval_on_selector_all(
            "a[href]", "els => els.slice(0, 60).map(e => e.innerText.trim() + ' -> ' + e.href)"
        )
        return "\n".join(i for i in items if i.strip().startswith(("http", "/")) or " -> " in i)

    return kit


def make_web_agent(llm: LLMClient, browser: Browser):
    try:
        import playwright  # noqa: F401

        toolkit, prompt = make_web_toolkit(browser), WEB_PROMPT
    except ImportError:
        toolkit = ToolKit()
        prompt = (
            "Browsing tools are not installed on this machine. Tell the user to run "
            "`uv sync --extra web && uv run playwright install chromium` to enable web navigation."
        )
    return make_tool_agent(
        "web", llm, "general", prompt, toolkit,
        force_first_tool="search" if toolkit.tools else None,
    )
