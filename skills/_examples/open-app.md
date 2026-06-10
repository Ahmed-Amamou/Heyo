---
name: open-windows-app
description: How to open Windows applications from Heyo (calculator, notepad, browser...)
agent: apps
triggers: open app, launch, start program, ouvre
---

# Opening Windows applications

Heyo runs inside WSL2, so Windows apps are launched through interop:

- Store/system apps by name: `cmd.exe /c start <name>` — e.g. `cmd.exe /c start calc`,
  `cmd.exe /c start notepad`.
- URLs or files with the default app: `cmd.exe /c start <url-or-path>` —
  e.g. `cmd.exe /c start https://example.com`.
- Specific executables: `powershell.exe -Command "Start-Process '<exe name or full path>'"`.
- A Windows Explorer window at a folder: `explorer.exe <windows-path>`.

Known apps:
| ask for | command |
|---|---|
| calculator | `cmd.exe /c start calc` |
| notepad | `cmd.exe /c start notepad` |
| browser / internet | `cmd.exe /c start https://www.google.com` |

If the user asks for an app not listed here, try `cmd.exe /c start <appname>` first.
