[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [Parameter(Mandatory = $true)]
    [string]$AccountEmail,

    [string]$ConfigDir = (Join-Path (Get-Location) ".gcloud-config-l4")
)

$ErrorActionPreference = "Stop"

function Invoke-GCloud {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed: gcloud $($Arguments -join ' ')"
    }
}

$resolvedConfigDir = [System.IO.Path]::GetFullPath($ConfigDir)
New-Item -ItemType Directory -Force -Path $resolvedConfigDir | Out-Null
$env:CLOUDSDK_CONFIG = $resolvedConfigDir

Invoke-GCloud -Arguments @("auth", "login", $AccountEmail)
Invoke-GCloud -Arguments @("config", "set", "project", $ProjectId)
Invoke-GCloud -Arguments @("auth", "application-default", "login", $AccountEmail)
Invoke-GCloud -Arguments @(
    "services",
    "enable",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "serviceusage.googleapis.com",
    "--project=$ProjectId"
)

Write-Host ""
Write-Host "Repo-local gcloud context is ready."
Write-Host "CLOUDSDK_CONFIG=$resolvedConfigDir"
Write-Host ""
Write-Host "Quick checks:"
Write-Host "  gcloud config get-value account"
Write-Host "  gcloud config get-value project"
Write-Host "  gcloud compute regions describe us-central1"
