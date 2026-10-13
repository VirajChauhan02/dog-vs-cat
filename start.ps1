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
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'model.keras'))) {
    $trainFolder = Join-Path $PSScriptRoot 'data\train'
    $hasLabeledImages = $false
    if (Test-Path -LiteralPath $trainFolder) {
        $hasLabeledImages = [bool](Get-ChildItem -LiteralPath $trainFolder -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(cat|dog)\.' } | Select-Object -First 1)
    }
    if (-not $hasLabeledImages) {
        & $venvPython download_data.py
    }
    & $venvPython train_model.py
}

& $venvPython -m streamlit run app.py

