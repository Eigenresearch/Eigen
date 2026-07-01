# Post-install PowerShell script for Eigen
# Called by Inno Setup after files are copied

param(
    [string]$InstallDir = ""
)

$ErrorActionPreference = "SilentlyContinue"

if ($InstallDir -eq "") {
    $InstallDir = "$env:PROGRAMFILES\Eigen"
}

Write-Host "[Eigen Post-Install] Configuring environment..."

# Add to PATH if not already present
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($currentPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$InstallDir", "Machine")
    Write-Host "[Eigen Post-Install] Added $InstallDir to system PATH (available after restart)."
}

# Register .eig file association
$extKey = "HKLM:\SOFTWARE\Classes\.eig"
if (-not (Test-Path $extKey)) {
    New-Item -Path $extKey -Force | Out-Null
    Set-ItemProperty -Path $extKey -Name "(Default)" -Value "EigenSourceFile"
    Write-Host "[Eigen Post-Install] Registered .eig file extension."
}

$typeKey = "HKLM:\SOFTWARE\Classes\EigenSourceFile"
if (-not (Test-Path $typeKey)) {
    New-Item -Path $typeKey -Force | Out-Null
    Set-ItemProperty -Path $typeKey -Name "(Default)" -Value "Eigen Source File"

    $iconKey = "$typeKey\DefaultIcon"
    New-Item -Path $iconKey -Force | Out-Null
    Set-ItemProperty -Path $iconKey -Name "(Default)" -Value "$InstallDir\eigen.exe,0"

    $cmdKey = "$typeKey\shell\open\command"
    New-Item -Path $cmdKey -Force | Out-Null
    Set-ItemProperty -Path $cmdKey -Name "(Default)" -Value "`"$InstallDir\eigen.exe`" run `"%1`""
    Write-Host "[Eigen Post-Install] Registered Eigen as handler for .eig files."
}

# Verify installation
if (Test-Path "$InstallDir\eigen.exe") {
    Write-Host "[Eigen Post-Install] Installation verified at $InstallDir"
    & "$InstallDir\eigen.exe" doctor
} else {
    Write-Warning "[Eigen Post-Install] eigen.exe not found at $InstallDir"
}

Write-Host "[Eigen Post-Install] Done."
