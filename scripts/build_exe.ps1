param(
    [ValidateSet("all", "label", "detect", "cli")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

switch ($Target) {
    "all" {
        $entry = "src\ai_labeller\app_all.py"
        $name = "GeckoAI-All"
    }
    "label" {
        $entry = "src\ai_labeller\app_label.py"
        $name = "GeckoAI-Label"
    }
    "detect" {
        $entry = "src\ai_labeller\app_detect.py"
        $name = "GeckoAI-Detect"
        $console = $false
    }
    "cli" {
        $entry = "src\ai_labeller\cli.py"
        $name = "GeckoAI-CLI"
        $console = $true
    }
}

$desktopIcon = "C:\Users\james\Desktop\app_icon.png"
$iconPath = if (Test-Path $desktopIcon) { $desktopIcon } else { "src\ai_labeller\assets\app_icon.png" }
$addData = @(
    "src\ai_labeller\assets;ai_labeller\assets",
    "src\ai_labeller\models;ai_labeller\models"
)

$cmd = @(
    "pyinstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--name", $name,
    "--paths", "src",
    "--icon", $iconPath
)

if (-not $console) {
    $cmd += "--noconsole"
}

foreach ($d in $addData) {
    $cmd += @("--add-data", $d)
}

$cmd += $entry

Write-Host "Building $name from $entry"
& $cmd[0] $cmd[1..($cmd.Length - 1)]
Write-Host "Done. Output: dist\$name\$name.exe"
