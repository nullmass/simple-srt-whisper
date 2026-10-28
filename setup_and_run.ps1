param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('cpu', 'cuda', 'remove-cuda')]
    [string]$Mode,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$venvDir = Join-Path $projectDir '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

function Stop-Setup {
    param([string]$Message)
    Write-Host "`nERROR: $Message" -ForegroundColor Red
    if (-not [Console]::IsInputRedirected) { [void](Read-Host 'Press Enter to close') }
    exit 1
}

function Test-PythonProbe {
    param([string]$Executable, [string[]]$Arguments)
    # Missing dependencies are expected. Windows PowerShell must not stop on native stderr.
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments *> $null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-Python {
    param([string[]]$Arguments)
    $ErrorActionPreference = 'Continue'
    & $script:venvPython @Arguments | Out-Host
    return $LASTEXITCODE
}

Write-Host "Listen setup ($Mode mode; shared .venv)"
if ($Mode -eq 'remove-cuda' -and -not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'No shared environment exists; nothing to remove.'
    exit 0
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonCandidates = @()
    foreach ($version in @('3.13', '3.12', '3.11', '3.10', '3.14')) {
        $pythonCandidates += ,@('py', "-$version")
    }
    $pythonCandidates += ,@('python')
    $pythonCommand = $null
    $versionProbe = 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)'
    foreach ($candidate in $pythonCandidates) {
        if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
        $probeArgs = @($candidate | Select-Object -Skip 1) + @('-c', $versionProbe)
        if (Test-PythonProbe $candidate[0] $probeArgs) {
            $pythonCommand = $candidate
            break
        }
    }
    if (-not $pythonCommand) { Stop-Setup 'Python 3.10 through 3.14 was not found.' }
    Write-Host "Creating shared .venv using $($pythonCommand -join ' ')..."
    $createArgs = @($pythonCommand | Select-Object -Skip 1) + @('-m', 'venv', $venvDir)
    & $pythonCommand[0] @createArgs
    if ($LASTEXITCODE -ne 0) { Stop-Setup 'Could not create .venv.' }
}

if ($Mode -eq 'remove-cuda') {
    Write-Host 'Removing optional NVIDIA runtimes and legacy PyTorch from .venv...'
    $code = Invoke-Python @('-m', 'pip', 'uninstall', '-y', 'nvidia-cudnn-cu12', 'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12', 'torch', 'torchvision', 'torchaudio')
    if ($code -ne 0) { Stop-Setup 'Could not remove GPU packages.' }
    Write-Host 'GPU packages removed. CPU mode remains available.'
    exit 0
}

Write-Host "`nChecking FFmpeg..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectDir 'install_ffmpeg.ps1')
if ($LASTEXITCODE -ne 0) { Stop-Setup 'FFmpeg setup failed.' }

$env:PATH = (Join-Path $venvDir 'Lib\site-packages\nvidia\cudnn\bin') + ';' +
    (Join-Path $venvDir 'Lib\site-packages\nvidia\cublas\bin') + ';' +
    (Join-Path $venvDir 'Lib\site-packages\nvidia\cuda_runtime\bin') + ';' +
    (Join-Path $projectDir 'tools\ffmpeg\bin') + ';' + $env:PATH

if (-not (Test-PythonProbe $venvPython @('-c', "import faster_whisper, rich, av; assert int(av.__version__.split('.')[0]) < 19"))) {
    Write-Host "`nInstalling shared Python packages..."
    $code = Invoke-Python @('-m', 'pip', 'install', '-r', (Join-Path $projectDir 'req.txt'))
    if ($code -ne 0) { Stop-Setup 'Could not install Python packages.' }
}

if ($Mode -eq 'cuda') {
    $runtimeProbe = "import ctypes, sys; from pathlib import Path; root = Path(sys.prefix) / 'Lib/site-packages/nvidia'; ctypes.WinDLL(str(root / 'cublas/bin/cublas64_12.dll')); ctypes.WinDLL(str(root / 'cudnn/bin/cudnn64_9.dll'))"
    # Check installed distributions as well as DLLs, avoiding reliance on loose project DLLs.
    $packageProbe = "from importlib.metadata import version; version('nvidia-cublas-cu12'); version('nvidia-cudnn-cu12')"
    if (-not (Test-PythonProbe $venvPython @('-c', $packageProbe)) -or
        -not (Test-PythonProbe $venvPython @('-c', $runtimeProbe))) {
        Write-Host "`nInstalling optional CUDA 12 and cuDNN 9 runtimes (no PyTorch)..."
        $code = Invoke-Python @('-m', 'pip', 'install', '-r', (Join-Path $projectDir 'req-cuda.txt'))
        if ($code -ne 0) { Stop-Setup 'Could not install CUDA runtimes. CPU mode is still available.' }
    }
    if (-not (Test-PythonProbe $venvPython @('-c', $runtimeProbe))) {
        Stop-Setup 'CUDA runtime DLLs could not be loaded. CPU mode is still available.'
    }
    if (-not (Test-PythonProbe $venvPython @('-c', 'import ctranslate2; assert ctranslate2.get_cuda_device_count() > 0'))) {
        Stop-Setup 'No CUDA device is available. Check the NVIDIA driver, or use start_cpu.bat.'
    }
}

Write-Host "`nStarting Listen on $Mode. Models download automatically if needed."
$code = Invoke-Python (@((Join-Path $projectDir 'main.py')) + $AppArgs + @('--device', $Mode))
if (-not [Console]::IsInputRedirected) { [void](Read-Host 'Press Enter to close') }
exit $code
