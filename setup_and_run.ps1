param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('cpu', 'cuda')]
    [string]$Mode,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$pythonCommand = $null

function Stop-Setup {
    param([string]$Message)
    Write-Host "`nERROR: $Message" -ForegroundColor Red
    [void](Read-Host 'Press Enter to close')
    exit 1
}

function Invoke-SelectedPython {
    param([string[]]$Arguments)
    if ($script:pythonCommand.Count -eq 1) {
        & $script:pythonCommand[0] @Arguments | Out-Host
    }
    else {
        & $script:pythonCommand[0] $script:pythonCommand[1] @Arguments | Out-Host
    }
    return $LASTEXITCODE
}

$pythonCandidates = @()
$pythonCandidates += ,@('python')
foreach ($version in @('3.14', '3.13', '3.12', '3.11', '3.10')) {
    $pythonCandidates += ,@('py', "-$version")
}

$versionProbe = 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)'
foreach ($candidate in $pythonCandidates) {
    if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) {
        continue
    }
    if ($candidate.Count -eq 1) {
        & $candidate[0] -c $versionProbe 2>$null
    }
    else {
        & $candidate[0] $candidate[1] -c $versionProbe 2>$null
    }
    if ($LASTEXITCODE -eq 0) {
        $pythonCommand = $candidate
        break
    }
}

if (-not $pythonCommand) {
    Stop-Setup 'Python 3.10 through 3.14 was not found. Install a supported Python version and run this file again.'
}

$pythonLabel = $pythonCommand -join ' '
Write-Host "Listen setup ($Mode mode)"
Write-Host "Using Python: $pythonLabel"

$venvName = if ($Mode -eq 'cpu') { '.venv-cpu' } else { '.venv-cuda' }
$venvDir = Join-Path $projectDir $venvName
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating $venvName..."
    $code = Invoke-SelectedPython @('-m', 'venv', $venvDir)
    if ($code -ne 0) {
        $code = Invoke-SelectedPython @('-m', 'ensurepip', '--upgrade')
        if ($code -eq 0) {
            $code = Invoke-SelectedPython @('-m', 'venv', $venvDir)
        }
    }
    if ($code -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        Stop-Setup 'Could not create the virtual environment.'
    }
}

Write-Host "`nChecking FFmpeg..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectDir 'install_ffmpeg.ps1')
if ($LASTEXITCODE -ne 0) {
    Stop-Setup 'FFmpeg setup failed. Check your internet connection and try again.'
}

$env:PATH = (Join-Path $venvDir 'Lib\site-packages\nvidia\cudnn\bin') + ';' +
    (Join-Path $venvDir 'Lib\site-packages\nvidia\cublas\bin') + ';' +
    (Join-Path $projectDir 'tools\ffmpeg\bin') + ';' + $env:PATH

& $venvPython -c "import faster_whisper, rich, av; raise SystemExit(0 if int(av.__version__.split('.')[0]) < 19 else 1)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nInstalling shared Python packages... This may take several minutes."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Stop-Setup 'Could not update pip.'
    }
    & $venvPython -m pip install -r (Join-Path $projectDir 'req.txt')
    if ($LASTEXITCODE -ne 0) {
        Stop-Setup 'Could not install the shared Python packages.'
    }
}

if ($Mode -eq 'cpu') {
    & $venvPython -c "import torch; raise SystemExit(0 if torch.version.cuda is None else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nInstalling CPU-only PyTorch..."
        & $venvPython -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) {
            Stop-Setup 'Could not install the CPU-only PyTorch build.'
        }
    }
}
else {
    & $venvPython -c "import torch; raise SystemExit(0 if torch.version.cuda is not None else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nInstalling CUDA 12.8 PyTorch and its runtime libraries..."
        & $venvPython -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) {
            Stop-Setup 'Could not install the CUDA PyTorch build.'
        }
    }

    & $venvPython -c "import torch, ctranslate2; assert torch.cuda.is_available(); assert ctranslate2.get_cuda_device_count() > 0" *> $null
    if ($LASTEXITCODE -ne 0) {
        Stop-Setup 'CUDA is not ready. Check that an NVIDIA GPU and a compatible NVIDIA driver are installed, then run start_cuda.bat again.'
    }
}

Write-Host "`nStarting Listen. The selected Whisper model downloads automatically if needed."
& $venvPython (Join-Path $projectDir 'main.py') @AppArgs
$appExitCode = $LASTEXITCODE
[void](Read-Host 'Press Enter to close')
exit $appExitCode
