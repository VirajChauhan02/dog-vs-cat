$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if ($python) {
        & py -3.11 -m venv .venv
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            throw 'Python 3.11 is required. Install it, then run start.ps1 again.'
        }
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        throw 'Could not create the project virtual environment. Check that Python 3.11 is installed.'
    }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip.' }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the project requirements.' }
}

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'model.keras'))) {
    $trainFolder = Join-Path $PSScriptRoot 'data\train'
    $hasLabeledImages = $false
    if (Test-Path -LiteralPath $trainFolder) {
        $hasLabeledImages = [bool](Get-ChildItem -LiteralPath $trainFolder -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(cat|dog)\.' } | Select-Object -First 1)
    }
    if (-not $hasLabeledImages) {
        & $venvPython download_data.py
        if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed. Read the error above, fix it, and rerun start.ps1.' }
    }
    & $venvPython train_model.py
    if ($LASTEXITCODE -ne 0) { throw 'Model training failed. Read the error above, fix it, and rerun start.ps1.' }
}

& $venvPython -m streamlit run app.py

