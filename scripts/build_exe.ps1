param(
    [ValidateSet("all", "label", "detect")]
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
    "--noconsole",
    "--name", $name,
    "--paths", "src",
    "--icon", $iconPath
)

foreach ($d in $addData) {
    $cmd += @("--add-data", $d)
}

$cmd += $entry

Write-Host "Building $name from $entry"
& $cmd[0] $cmd[1..($cmd.Length - 1)]
Write-Host "Done. Output: dist\$name\$name.exe"
