$ErrorActionPreference = 'Stop'
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
Copy-Item .env.example .env -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe -m alembic upgrade head
& .\.venv\Scripts\python.exe -m scripts.bootstrap --skip-download
Write-Host 'Setup complete. Edit .env, then run scripts/bootstrap.py to obtain historical data.'
