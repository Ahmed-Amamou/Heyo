# Heyo voice client — Windows setup (run in PowerShell):
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1              # install deps + run
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Startup     # + auto-start at login
#
# The client is thin (mic + speaker + wake word + hotkey); speech-to-text and
# text-to-speech run on the Heyo server inside WSL, reachable at localhost.
param(
    [switch]$Startup,
    [string]$Server = "http://localhost:8000",
    [string]$Hotkey = "ctrl+alt+h"
)
$ErrorActionPreference = "Stop"

# -- find Python (3.9+)
$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) { $python = "py" }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $python = "python" }
if (-not $python -or (& $python -c "import sys; print(sys.version_info >= (3, 9))") -ne "True") {
    Write-Host "Python 3.9+ not found. Install it, then re-run:" -ForegroundColor Yellow
    Write-Host "    winget install Python.Python.3.12"
    exit 1
}

Write-Host "Installing client deps (numpy, sounddevice, httpx, vosk, keyboard)..."
& $python -m pip install --user --quiet --upgrade numpy sounddevice httpx vosk keyboard

# -- copy the client out of WSL so it survives reboots / WSL being down
$dest = Join-Path $env:LOCALAPPDATA "Heyo"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item (Join-Path $PSScriptRoot "client.py") $dest -Force
Write-Host "Client installed to $dest\client.py"

if ($Startup) {
    # hidden-console autostart: pythonw + a shortcut in shell:startup
    $pyw = & $python -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))"
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Startup")) "Heyo Voice.lnk"))
    $lnk.TargetPath = $pyw
    $lnk.Arguments = "`"$dest\client.py`" --server $Server --hotkey `"$Hotkey`""
    $lnk.WorkingDirectory = $dest
    $lnk.Save()
    Write-Host "Auto-start enabled (shortcut in shell:startup). It waits quietly until the server is up."
}

Write-Host "`nStarting Heyo — say `"Heyo`" or press $Hotkey ..." -ForegroundColor Green
& $python (Join-Path $dest "client.py") --server $Server --hotkey $Hotkey
