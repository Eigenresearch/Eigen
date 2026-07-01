# Uninstall cleanup PowerShell script for Eigen
# Called by Inno Setup during uninstallation

param(
    [string]$InstallDir = "$env:PROGRAMFILES\Eigen"
)

$ErrorActionPreference = "SilentlyContinue"

Write-Host "[Eigen Uninstall] Cleaning up environment..."

# Remove from PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($currentPath -like "*$InstallDir*") {
    $newPath = $currentPath -replace [regex]::Escape(";$InstallDir"), "" -replace [regex]::Escape("$InstallDir;"), "" -replace [regex]::Escape($InstallDir), ""
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
    Write-Host "[Eigen Uninstall] Removed $InstallDir from system PATH."
}

# Remove .eig file association
Remove-Item -Path "HKLM:\SOFTWARE\Classes\.eig" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "HKLM:\SOFTWARE\Classes\EigenSourceFile" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "[Eigen Uninstall] Removed .eig file association."

# Remove context menu entries
Remove-Item -Path "HKCU:\Software\Classes\*\shell\OpenWithEigen" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "HKCU:\Software\Classes\Directory\shell\OpenWithEigen" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "[Eigen Uninstall] Removed context menu entries."

Write-Host "[Eigen Uninstall] Cleanup complete."
