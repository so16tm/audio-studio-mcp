<#
.SYNOPSIS
Windows / NVIDIA 用の ACE-Step 1.5 + MOSS-SoundEffect v2 環境を構築します。
.DESCRIPTION
固定した公式ソースと重みを取得し、Python 3.12 の .venv-hybrid に導入します。
既存の .venv / .venv-cuda と MCP クライアント設定は変更しません。
uv がない場合は winget install astral-sh.uv で導入してから再実行してください。
.PARAMETER Python
Python 3.12 の実行ファイル、または uv の Python 指定（既定 3.12）。
.EXAMPLE
powershell -ExecutionPolicy Bypass -File install-hybrid.ps1 -Python 3.12
#>
[CmdletBinding()]
param([string]$Python = "3.12", [switch]$Help)

# PowerShell 自体の失敗は直ちに停止し、ネイティブコマンドは終了コードも確認する。
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
if ($Help) { Get-Help $PSCommandPath -Detailed; exit 0 }
function Die($Message) { Write-Host "ERROR: $Message" -ForegroundColor Red; exit 1 }
function Check($Operation) { if ($LASTEXITCODE -ne 0) { Die "$Operation failed (exit $LASTEXITCODE)." } }
function Have($Command) { return [bool](Get-Command $Command -ErrorAction SilentlyContinue) }
function Have-Commit($Destination, $Commit) {
    # cat-file の失敗は未取得を表す。PS 5.1 はリダイレクトした stderr を
    # PowerShell エラーへ変換するため、この存在確認だけ局所的に許容する。
    $ErrorActionPreference = "Continue"
    git -C $Destination cat-file -e "$Commit^{commit}" 2>$null
    return $LASTEXITCODE -eq 0
}

if ($env:OS -ne "Windows_NT" -or -not [Environment]::Is64BitOperatingSystem) { Die "64-bit Windows is required." }
foreach ($Command in @("git", "uv", "nvidia-smi")) {
    if (-not (Have $Command)) {
        Die "$Command is required. Install git (winget install Git.Git), uv (winget install astral-sh.uv), and the NVIDIA driver before running this installer."
    }
}
git --version; Check "git"
uv --version; Check "uv"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; Check "NVIDIA GPU check"
$Ffmpeg = if ($env:AUDIO_STUDIO_FFMPEG) { $env:AUDIO_STUDIO_FFMPEG } else { "ffmpeg" }
if (-not (Have $Ffmpeg)) { Die "ffmpeg is required: winget install Gyan.FFmpeg; or set AUDIO_STUDIO_FFMPEG to its executable path." }
& $Ffmpeg -version | Select-Object -First 1
Check "ffmpeg"

$RepoDir = $PSScriptRoot
function Get-PinnedSource($Name, $Url, $Commit) {
    $Destination = Join-Path $RepoDir "vendor\$Name"
    $NewClone = $false
    if (-not (Test-Path -LiteralPath $Destination)) {
        New-Item -ItemType Directory -Force -Path (Join-Path $RepoDir "vendor") -ErrorAction Stop | Out-Null
        git clone --filter=blob:none --no-checkout $Url $Destination
        Check "clone $Name"
        $NewClone = $true
    } elseif (-not (Test-Path -LiteralPath (Join-Path $Destination ".git"))) {
        Die "$Destination already exists and is not a git checkout. Move it aside yourself before retrying."
    }
    $Origin = (git -C $Destination remote get-url origin) -join ""
    Check "read $Name origin"
    if ($Origin.Trim() -ne $Url) { Die "$Destination has another origin; leaving it unchanged." }
    if (-not $NewClone) {
        $Status = git -C $Destination status --porcelain
        Check "check $Name changes"
        if ($Status) { Die "$Destination has local changes; leaving it unchanged. Preserve them before retrying." }
    }
    if (-not (Have-Commit $Destination $Commit)) {
        git -C $Destination fetch origin $Commit
        Check "fetch $Name"
    }
    git -C $Destination -c advice.detachedHead=false checkout --quiet $Commit
    Check "checkout $Name"
}

Get-PinnedSource "MOSS-TTS" "https://github.com/OpenMOSS/MOSS-TTS.git" "934d6826b084c46a0d033402174d5f8ac4ed2519"
Get-PinnedSource "ACE-Step-1.5" "https://github.com/ace-step/ACE-Step-1.5.git" "ca1e85fe9430179831e6bc6be790c332190a3866"

$EnvironmentDir = Join-Path $RepoDir ".venv-hybrid"
$Py = Join-Path $EnvironmentDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $EnvironmentDir)) {
    uv venv --python $Python $EnvironmentDir
    Check "create Python 3.12 environment"
} elseif (-not (Test-Path -LiteralPath $Py)) {
    Die "$EnvironmentDir already exists without Python; leaving it unchanged."
}
& $Py -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'; print(sys.version)"
Check "Python 3.12 check"

$env:PYTHONIOENCODING = "utf-8"
Push-Location $RepoDir
try {
    # ACE はソースから利用する。uv sync や SA3 の cuda extra は併用しない。
    uv pip install --python $Py --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -e "vendor/MOSS-TTS/moss_soundeffect_v2[torch-cu128]" "vector-quantize-pytorch==1.32.1" "loguru==0.7.3" "accelerate==1.15.0" "hf-xet>=1.7" -e .
    Check "install hybrid dependencies"
    # Python 3.12 の .pth は Windows のコードページで読まれるため、
    # editable hook のパスだけを ASCII の相対パスへ置換する。
    $Site = Join-Path $EnvironmentDir "Lib\site-packages"
    Get-ChildItem -LiteralPath $Site -Filter "_editable_impl_audio_studio_mcp*.pth" | ForEach-Object {
        $Line = [IO.File]::ReadAllText($_.FullName, [Text.Encoding]::UTF8).Trim()
        if ($Line -match '[^\x00-\x7F]') { [IO.File]::WriteAllText($_.FullName, "..\..\..\src`r`n", [Text.Encoding]::ASCII) }
    }
    & $Py -c "import audio_studio_mcp, torch; assert torch.cuda.is_available(), 'CUDA GPU is unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
    Check "runtime / CUDA check"
    & $Py (Join-Path $RepoDir "scripts\download_hybrid_weights.py")
    Check "download complete pinned model snapshots"
} finally { Pop-Location }

$Executable = Join-Path $EnvironmentDir "Scripts\audio-studio-mcp.exe"
Write-Host "Hybrid setup complete. MCP registration is a separate step (README.ja.md)."
Write-Host "claude mcp add --scope user audio-studio --env AUDIO_STUDIO_BACKEND=ace-moss -- `"$Executable`""
Write-Host "codex mcp add audio-studio --env AUDIO_STUDIO_BACKEND=ace-moss -- `"$Executable`""
