$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not $env:DEEPSEEK_API_KEY) {
    $secureKey = Read-Host "DeepSeek API key" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try { $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
}
if (-not $env:DEEPSEEK_API_KEY) { throw "DeepSeek API key is required" }
$env:AGENT_LLM_PROVIDER = "deepseek"
$config = Join-Path $PSScriptRoot "mcp.json"
try {
    if (Test-Path -LiteralPath $config) {
        & $python -X utf8 (Join-Path $PSScriptRoot "web_ui.py") --root $PSScriptRoot --mcp-config $config
    } else {
        & $python -X utf8 (Join-Path $PSScriptRoot "web_ui.py") --root $PSScriptRoot
    }
} finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:AGENT_LLM_PROVIDER -ErrorAction SilentlyContinue
}
