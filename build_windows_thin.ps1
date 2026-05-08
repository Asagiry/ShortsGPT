$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$appName = "AI Shorts Studio"
$tempBuild = Join-Path $repo "build_root"
$tempDist = Join-Path $repo "dist_root"
$builtApp = Join-Path $tempDist $appName
$targetExe = Join-Path $repo "$appName.exe"
$targetInternal = Join-Path $repo "_internal"

Write-Host "Building UI..."
Push-Location (Join-Path $repo "ui")
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw "UI build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "Closing old app processes..."
$targets = Get-CimInstance Win32_Process | Where-Object {
    (
        ($_.ExecutablePath -like "$repo\*") -or
        ($_.CommandLine -like "*AI-Youtube-Shorts-Generator*") -or
        ($_.CommandLine -like "*AI Shorts Studio*")
    ) -and
    $_.ProcessId -ne $PID -and
    $_.Name -notlike "powershell.exe" -and
    $_.Name -notlike "pwsh.exe"
}
$targets | ForEach-Object {
    if ($_.ProcessId) {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Cleaning previous build folders..."
Remove-Item -LiteralPath $tempBuild -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $tempDist -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Building desktop app..."
& .\.venv\Scripts\python.exe -m PyInstaller `
    -y `
    --noconsole `
    --onedir `
    --name $appName `
    --workpath $tempBuild `
    --distpath $tempDist `
    --add-data "ui\dist;ui\dist" `
    --add-data ".venv\Lib\site-packages\ctranslate2;ctranslate2" `
    --add-data ".venv\Lib\site-packages\av.libs;av.libs" `
    --add-data ".venv\Lib\site-packages\numpy.libs;numpy.libs" `
    --add-data ".venv\Lib\site-packages\onnxruntime\capi;onnxruntime\capi" `
    --collect-all yt_dlp `
    --hidden-import yt_dlp `
    --hidden-import webview `
    --hidden-import clr `
    --hidden-import clr_loader `
    --hidden-import wave `
    --hidden-import audioop `
    --hidden-import chunk `
    --hidden-import struct `
    --hidden-import math `
    --hidden-import urllib.request `
    --hidden-import urllib.error `
    studio.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

Write-Host "Publishing app to repository root..."
if (-not (Test-Path -LiteralPath (Join-Path $builtApp "$appName.exe"))) {
    throw "Built exe was not found: $builtApp\$appName.exe"
}

Remove-Item -LiteralPath $targetExe -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $targetInternal -Recurse -Force -ErrorAction SilentlyContinue
Move-Item -LiteralPath (Join-Path $builtApp "$appName.exe") -Destination $targetExe
Move-Item -LiteralPath (Join-Path $builtApp "_internal") -Destination $targetInternal

Write-Host "Smoke testing imports..."
$env:AI_SHORTS_SMOKE_IMPORTS = "1"
try {
    & $targetExe
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed with exit code $LASTEXITCODE" }
}
finally {
    Remove-Item Env:\AI_SHORTS_SMOKE_IMPORTS -ErrorAction SilentlyContinue
}

Write-Host "Cleaning temp and old dist folders..."
Remove-Item -LiteralPath $tempBuild -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $tempDist -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $repo "dist_fixed2") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $repo "$appName.spec") -Force -ErrorAction SilentlyContinue

Write-Host "Done: $targetExe"
