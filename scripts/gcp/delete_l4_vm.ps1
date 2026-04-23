[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [Parameter(Mandatory = $true)]
    [string]$Zone,

    [string]$VmName = "falcon-pipeline-l4"
)

$ErrorActionPreference = "Stop"

function Invoke-GCloud {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowFailure
    )

    $output = & gcloud @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine

    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "gcloud failed: gcloud $($Arguments -join ' ')`n$text"
    }

    [pscustomobject]@{
        ExitCode = $exitCode
        Output = $text
    }
}

$deleteResult = Invoke-GCloud -Arguments @(
    "compute",
    "instances",
    "delete",
    $VmName,
    "--project=$ProjectId",
    "--zone=$Zone",
    "--quiet"
) -AllowFailure

if ($deleteResult.ExitCode -ne 0 -and $deleteResult.Output -notmatch "was not found") {
    throw "Instance delete failed.`n$($deleteResult.Output)"
}

$instanceCheck = Invoke-GCloud -Arguments @(
    "compute",
    "instances",
    "describe",
    $VmName,
    "--project=$ProjectId",
    "--zone=$Zone"
) -AllowFailure

if ($instanceCheck.ExitCode -eq 0 -or $instanceCheck.Output -notmatch "was not found") {
    throw "Instance verification failed. Expected NOT_FOUND for $VmName.`n$($instanceCheck.Output)"
}

$diskCheck = Invoke-GCloud -Arguments @(
    "compute",
    "disks",
    "describe",
    $VmName,
    "--project=$ProjectId",
    "--zone=$Zone"
) -AllowFailure

if ($diskCheck.ExitCode -eq 0 -or $diskCheck.Output -notmatch "was not found") {
    throw "Disk verification failed. Expected NOT_FOUND for boot disk $VmName.`n$($diskCheck.Output)"
}

Write-Host "Deleted and verified absence for instance and boot disk:"
Write-Host "PROJECT_ID=$ProjectId"
Write-Host "ZONE=$Zone"
Write-Host "VM_NAME=$VmName"
