$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$binDir = Join-Path $PSScriptRoot 'tools\ffmpeg\bin'
$ffmpeg = Join-Path $binDir 'ffmpeg.exe'
if (Test-Path -LiteralPath $ffmpeg) {
    exit 0
}

$workDir = Join-Path ([System.IO.Path]::GetTempPath()) ('listen-ffmpeg-' + [guid]::NewGuid().ToString('N'))
$archive = Join-Path $workDir 'ffmpeg.zip'
$extractDir = Join-Path $workDir 'extracted'

try {
    New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
    Write-Host 'Downloading FFmpeg essentials build from gyan.dev...'
    Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $extractDir -Force

    $downloadedFfmpeg = Get-ChildItem -LiteralPath $extractDir -Filter 'ffmpeg.exe' -File -Recurse | Select-Object -First 1
    if (-not $downloadedFfmpeg) {
        throw 'The downloaded archive did not contain ffmpeg.exe.'
    }

    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    Copy-Item -LiteralPath $downloadedFfmpeg.FullName -Destination $ffmpeg

    $downloadedFfprobe = Join-Path $downloadedFfmpeg.DirectoryName 'ffprobe.exe'
    if (Test-Path -LiteralPath $downloadedFfprobe) {
        Copy-Item -LiteralPath $downloadedFfprobe -Destination (Join-Path $binDir 'ffprobe.exe')
    }
    Write-Host 'FFmpeg is ready.'
}
finally {
    if (Test-Path -LiteralPath $workDir) {
        Remove-Item -LiteralPath $workDir -Recurse -Force
    }
}
