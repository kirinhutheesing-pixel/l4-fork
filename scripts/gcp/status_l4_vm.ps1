[CmdletBinding()]
param(
    [string]$ProjectId = $env:PROJECT_ID,
    [string]$Zone = $env:ZONE,
    [string]$VmName = $(if ($env:VM_NAME) { $env:VM_NAME } else { "falcon-pipeline-l4" }),
    [string]$ContainerName = $(if ($env:CONTAINER_NAME) { $env:CONTAINER_NAME } else { "falcon-pipeline-l4" }),
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8080 })
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

function Write-Section {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    Write-Host ""
    Write-Host "== $Title =="
}

function Invoke-Remote {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    if (-not $ProjectId -or -not $Zone) {
        return [pscustomobject]@{
            ExitCode = 1
            Output = "ProjectId and Zone are required for remote VM checks."
        }
    }

    Invoke-GCloud -Arguments @(
        "compute",
        "ssh",
        $VmName,
        "--project=$ProjectId",
        "--zone=$Zone",
        "--strict-host-key-checking=no",
        "--command=$Command"
    ) -AllowFailure
}

Write-Section "Local gcloud"
$account = Invoke-GCloud -Arguments @("config", "get-value", "account") -AllowFailure
$configuredProject = Invoke-GCloud -Arguments @("config", "get-value", "project") -AllowFailure
Write-Host "account=$($account.Output)"
Write-Host "configured_project=$($configuredProject.Output)"
Write-Host "target_project=$ProjectId"
Write-Host "target_zone=$Zone"
Write-Host "vm_name=$VmName"

Write-Section "Instance"
$instance = Invoke-GCloud -Arguments @(
    "compute",
    "instances",
    "describe",
    $VmName,
    "--project=$ProjectId",
    "--zone=$Zone",
    "--format=value(status,machineType,networkInterfaces[0].accessConfigs[0].natIP)"
) -AllowFailure
Write-Host $instance.Output

Write-Section "Boot Disk"
$disk = Invoke-GCloud -Arguments @(
    "compute",
    "disks",
    "describe",
    $VmName,
    "--project=$ProjectId",
    "--zone=$Zone",
    "--format=value(status,sizeGb,type)"
) -AllowFailure
Write-Host $disk.Output

Write-Section "Host GPU"
$hostGpu = Invoke-Remote -Command "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
Write-Host $hostGpu.Output

Write-Section "Docker GPU"
$dockerGpu = Invoke-Remote -Command "sudo docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader"
Write-Host $dockerGpu.Output

Write-Section "Container"
$container = Invoke-Remote -Command "sudo docker ps -a --filter name=$ContainerName --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
Write-Host $container.Output

Write-Section "Health"
$health = Invoke-Remote -Command "curl -fsS http://127.0.0.1:$Port/api/healthz"
Write-Host $health.Output

Write-Section "State Summary"
$state = Invoke-Remote -Command "curl -fsS http://127.0.0.1:$Port/api/state | python3 -c 'import json,sys; s=json.load(sys.stdin); print(json.dumps({""readiness"":s.get(""readiness""),""source"":s.get(""source""),""frame"":s.get(""frame""),""model_status"":s.get(""model_status""),""engines"":(s.get(""result"") or {}).get(""engines""),""scene_counts"":((s.get(""result"") or {}).get(""scene_annotations"") or {}).get(""counts"")}, indent=2))'"
Write-Host $state.Output

Write-Section "Recent Container Logs"
$logs = Invoke-Remote -Command "sudo docker logs --tail 80 $ContainerName 2>&1"
Write-Host $logs.Output
