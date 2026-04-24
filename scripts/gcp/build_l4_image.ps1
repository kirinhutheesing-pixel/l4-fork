[CmdletBinding()]
param(
    [string]$ProjectId = $env:PROJECT_ID,
    [string]$Zone = $env:ZONE,
    [string]$VmName = $(if ($env:VM_NAME) { $env:VM_NAME } else { "falcon-pipeline-l4" }),
    [string]$RepoUrl = $(if ($env:REPO_URL) { $env:REPO_URL } else { "https://github.com/kirinhutheesing-pixel/l4-fork.git" }),
    [string]$RemoteDir = $(if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "/home/kirin/l4-fork" }),
    [string]$ImageName = $(if ($env:IMAGE_NAME) { $env:IMAGE_NAME } else { "falcon-pipeline:l4" }),
    [string]$BuildLog = $(if ($env:BUILD_LOG) { $env:BUILD_LOG } else { "/tmp/falcon-pipeline-build.log" }),
    [string]$BuildPidFile = $(if ($env:BUILD_PID_FILE) { $env:BUILD_PID_FILE } else { "/tmp/falcon-pipeline-build.pid" }),
    [int]$PollSeconds = 30,
    [int]$TimeoutMinutes = 45,
    [switch]$NoIap
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    throw "ProjectId is required. Pass -ProjectId or set PROJECT_ID."
}
if (-not $Zone) {
    throw "Zone is required. Pass -Zone or set ZONE."
}

function ConvertTo-BashSingleQuoted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return "'" + $Value.Replace("'", "'\''") + "'"
}

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

function Invoke-RemoteBash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script,
        [switch]$AllowFailure
    )

    $arguments = @(
        "compute",
        "ssh",
        $VmName,
        "--project=$ProjectId",
        "--zone=$Zone",
        "--strict-host-key-checking=no"
    )
    if (-not $NoIap) {
        $arguments += "--tunnel-through-iap"
    }
    $arguments += "--command=bash -lc $(ConvertTo-BashSingleQuoted -Value $Script)"

    Invoke-GCloud -Arguments $arguments -AllowFailure:$AllowFailure
}

$repoUrlQuoted = ConvertTo-BashSingleQuoted -Value $RepoUrl
$remoteDirQuoted = ConvertTo-BashSingleQuoted -Value $RemoteDir
$imageNameQuoted = ConvertTo-BashSingleQuoted -Value $ImageName
$buildLogQuoted = ConvertTo-BashSingleQuoted -Value $BuildLog
$buildPidFileQuoted = ConvertTo-BashSingleQuoted -Value $BuildPidFile

Write-Host "Preparing checkout on $VmName ($ProjectId/$Zone)..."
$checkoutScript = @"
set -euo pipefail
if [ -d ${remoteDirQuoted}/.git ]; then
  cd ${remoteDirQuoted}
  git fetch --all --prune
  git pull --ff-only
elif [ -e ${remoteDirQuoted} ]; then
  echo "Remote path exists but is not a git checkout: ${RemoteDir}" >&2
  exit 2
else
  mkdir -p "`$(dirname ${remoteDirQuoted})"
  git clone ${repoUrlQuoted} ${remoteDirQuoted}
  cd ${remoteDirQuoted}
fi
printf 'commit='
git rev-parse --short HEAD
"@
$checkout = Invoke-RemoteBash -Script $checkoutScript
Write-Host $checkout.Output

Write-Host "Starting detached Docker build for $ImageName..."
$startBuildScript = @"
set -euo pipefail
if [ -f ${buildPidFileQuoted} ] && kill -0 "`$(cat ${buildPidFileQuoted})" >/dev/null 2>&1; then
  echo "Existing build is still running with pid=`$(cat ${buildPidFileQuoted})"
else
  cd ${remoteDirQuoted}
  : > ${buildLogQuoted}
  nohup sudo docker build -f Dockerfile.gcp-l4 -t ${imageNameQuoted} . > ${buildLogQuoted} 2>&1 < /dev/null &
  echo `$! > ${buildPidFileQuoted}
  echo "Started build pid=`$(cat ${buildPidFileQuoted})"
fi
"@
$start = Invoke-RemoteBash -Script $startBuildScript
Write-Host $start.Output

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds $PollSeconds
    $pollScript = @"
set -euo pipefail
if [ ! -f ${buildPidFileQuoted} ]; then
  echo "Build pid file missing: ${BuildPidFile}" >&2
  exit 3
fi
pid="`$(cat ${buildPidFileQuoted})"
if kill -0 "`$pid" >/dev/null 2>&1; then
  echo "BUILD_RUNNING pid=`$pid"
  tail -n 40 ${buildLogQuoted} || true
  exit 2
fi
if sudo docker image inspect ${imageNameQuoted} >/dev/null 2>&1; then
  echo "IMAGE_READY ${ImageName}"
  sudo docker image inspect ${imageNameQuoted} --format '{{.Id}} {{.Size}}'
  exit 0
fi
echo "BUILD_FAILED"
tail -n 120 ${buildLogQuoted} || true
exit 1
"@
    $poll = Invoke-RemoteBash -Script $pollScript -AllowFailure
    Write-Host $poll.Output

    if ($poll.ExitCode -eq 0 -and $poll.Output -match "IMAGE_READY") {
        Write-Host ""
        Write-Host "L4 image build completed."
        Write-Host "PROJECT_ID=$ProjectId"
        Write-Host "ZONE=$Zone"
        Write-Host "VM_NAME=$VmName"
        Write-Host "IMAGE_NAME=$ImageName"
        Write-Host "BUILD_LOG=$BuildLog"
        exit 0
    }

    if ($poll.ExitCode -ne 2) {
        throw "Detached Docker build failed. Check $BuildLog on the VM."
    }
}

throw "Detached Docker build did not complete within $TimeoutMinutes minutes. Check $BuildLog on the VM."
