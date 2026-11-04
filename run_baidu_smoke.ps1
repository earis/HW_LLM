$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$keyPath = Join-Path $root "BAIDU_KEY"
$python = "C:\pytorch_m\hw\hw2\gensim_text_classification_lab\llm_text\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $keyPath -PathType Leaf)) {
    throw "BAIDU_KEY was not found."
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The requested Python interpreter was not found: $python"
}

$key = (Get-Content -LiteralPath $keyPath -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($key)) {
    throw "BAIDU_KEY is empty."
}

$env:BAIDU_API_KEY = $key
$env:BAIDU_MODEL = "deepseek-v4.1-flash"
Set-Location -LiteralPath $root
& $python ".\src\run_experiment.py" --provider baidu --smoke
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
