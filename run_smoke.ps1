$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$keyPath = Join-Path $root "KIMI_KEY"
$python = "C:\pytorch_m\hw\hw2\gensim_text_classification_lab\llm_text\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $keyPath -PathType Leaf)) {
    throw "KIMI_KEY was not found."
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The requested Python interpreter was not found: $python"
}

$key = (Get-Content -LiteralPath $keyPath -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($key)) {
    throw "KIMI_KEY is empty."
}

$env:MOONSHOT_API_KEY = $key
$env:MOONSHOT_MODEL = "kimi-k3.1-pro"
Set-Location -LiteralPath $root
& $python ".\src\run_experiment.py" --smoke
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
