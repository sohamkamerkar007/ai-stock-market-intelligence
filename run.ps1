param([ValidateRange(1, 65535)][int]$Port = 8000)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Virtual environment missing. Run .\setup.ps1 first.'
}
if (-not (Test-Path -LiteralPath '.env')) {
    throw 'Configuration missing. Run .\setup.ps1 first.'
}
Write-Host "AI Stock Market Intelligence: http://127.0.0.1:$Port"
& $venvPython -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
if ($LASTEXITCODE -ne 0) { throw "Application exited with code $LASTEXITCODE" }
