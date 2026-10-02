param(
    [switch]$SkipData,
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    $launcher = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $launcher = @('py', '-3.11') }
    }
    if (-not $launcher -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $launcher = @('python') }
    }
    if (-not $launcher) { throw 'Python 3.11 is required. Install it from python.org, then rerun setup.ps1.' }
    Write-Host 'Creating Python 3.11 virtual environment...'
    if ($launcher.Count -eq 2) { & py -3.11 -m venv .venv } else { & python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

function Invoke-VenvPython {
    & $venvPython @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($args -join ' ')" }
}

Write-Host 'Installing project dependencies...'
Invoke-VenvPython -m pip install --upgrade pip
Invoke-VenvPython -m pip install -e '.[dev]'

if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Host 'Created .env with local SQLite defaults.'
} else {
    Write-Host 'Keeping existing .env configuration.'
}

Write-Host 'Initializing database...'
Invoke-VenvPython -m alembic upgrade head
if ($SkipData) {
    Invoke-VenvPython -m scripts.bootstrap --skip-download
    Write-Host 'Skipped market-data download. Run setup.ps1 without -SkipData before using all pages.'
} else {
    Write-Host 'Downloading actual Yahoo Finance daily history. This can take several minutes.'
    Invoke-VenvPython -m scripts.bootstrap --years 8 --strict
    if (-not $SkipModels) {
        Write-Host 'Training saved real-market direction, price, and volatility models...'
        Invoke-VenvPython -m scripts.train_real_direction
        Invoke-VenvPython -m scripts.train_final_stock_models
    }
}

Write-Host 'Setup complete. Start the app with .\run.ps1.'
