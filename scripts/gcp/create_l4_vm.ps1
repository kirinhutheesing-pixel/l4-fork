[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [string]$VmName = "falcon-pipeline-l4",

    [string[]]$Zones = @(
        "us-east4-c",
        "us-east1-c",
        "us-east1-b",
        "us-west4-c",
        "us-west4-a",
        "us-central1-a",
        "us-central1-b",
        "us-central1-c"
    ),

    [string]$MachineType = "g2-standard-8",
    [int]$BootDiskSizeGb = 200,
    [string]$BootDiskType = "pd-balanced",
    [string]$ImageFamily = "ubuntu-2204-lts",
    [string]$ImageProject = "ubuntu-os-cloud",
    [int]$BootstrapTimeoutMinutes = 20,
    [string]$StartupScriptPath = (Join-Path $PSScriptRoot "bootstrap_g2_ubuntu2204.sh")
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

function Invoke-RemoteCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Zone,
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [switch]$AllowFailure
    )

    Invoke-GCloud -Arguments @(
        "compute",
        "ssh",
        $VmName,
        "--project=$ProjectId",
        "--zone=$Zone",
        "--strict-host-key-checking=no",
        "--command=$Command"
    ) -AllowFailure:$AllowFailure
}

function Test-ZoneExhausted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $markers = @(
        "ZONE_RESOURCE_POOL_EXHAUSTED",
        "resource pool exhausted",
        "does not have enough resources available",
        "currently unavailable in the zone"
    )

    foreach ($marker in $markers) {
        if ($Message -match [Regex]::Escape($marker)) {
            return $true
        }
    }

    return $false
}

function Wait-ForBootstrap {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Zone
    )

    $deadline = (Get-Date).AddMinutes($BootstrapTimeoutMinutes)
    while ((Get-Date) -lt $deadline) {
        $probe = Invoke-RemoteCommand -Zone $Zone -Command "sudo test -f /var/lib/falcon-pipeline/bootstrap.ready && echo READY" -AllowFailure
        if ($probe.ExitCode -eq 0 -and $probe.Output -match "READY") {
            return
        }
        Start-Sleep -Seconds 15
    }

    throw "Bootstrap did not finish within $BootstrapTimeoutMinutes minutes."
}

$resolvedStartupScriptPath = [System.IO.Path]::GetFullPath($StartupScriptPath)
if (-not (Test-Path -LiteralPath $resolvedStartupScriptPath)) {
    throw "Startup script not found: $resolvedStartupScriptPath"
}

$successfulZone = $null

foreach ($zone in $Zones) {
    Write-Host "Trying zone $zone..."
    $result = Invoke-GCloud -Arguments @(
        "compute",
        "instances",
        "create",
        $VmName,
        "--project=$ProjectId",
        "--zone=$zone",
        "--machine-type=$MachineType",
        "--boot-disk-size=${BootDiskSizeGb}GB",
        "--boot-disk-type=$BootDiskType",
        "--image-family=$ImageFamily",
        "--image-project=$ImageProject",
        "--maintenance-policy=TERMINATE",
        "--boot-disk-auto-delete",
        "--no-shielded-secure-boot",
        "--metadata=serial-port-enable=TRUE",
        "--metadata-from-file=startup-script=$resolvedStartupScriptPath"
    ) -AllowFailure

    if ($result.ExitCode -eq 0) {
        $successfulZone = $zone
        break
    }

    if (Test-ZoneExhausted -Message $result.Output) {
        Write-Warning "Zone $zone is exhausted for this shape. Trying the next zone."
        continue
    }

    throw "VM creation failed in zone $zone.`n$($result.Output)"
}

if (-not $successfulZone) {
    throw "Could not allocate $VmName in any configured zone."
}

Write-Host "Created $VmName in $successfulZone. Waiting for bootstrap..."
Wait-ForBootstrap -Zone $successfulZone

Write-Host "Running host GPU check..."
$hostGpuCheck = Invoke-RemoteCommand -Zone $successfulZone -Command "nvidia-smi"
Write-Host $hostGpuCheck.Output

Write-Host "Running Docker GPU check..."
$dockerGpuCheck = Invoke-RemoteCommand -Zone $successfulZone -Command "sudo docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi"
Write-Host $dockerGpuCheck.Output

Write-Host ""
Write-Host "L4 VM is ready."
Write-Host "PROJECT_ID=$ProjectId"
Write-Host "VM_NAME=$VmName"
Write-Host "ZONE=$successfulZone"
Write-Host ""
Write-Host "Next:"
Write-Host "  gcloud compute ssh $VmName --project=$ProjectId --zone=$successfulZone --strict-host-key-checking=no"
