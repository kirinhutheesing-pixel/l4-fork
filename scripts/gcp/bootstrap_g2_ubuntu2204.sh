#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/var/log/falcon-pipeline-bootstrap.log"
STATE_DIR="/var/lib/falcon-pipeline"
READY_FILE="${STATE_DIR}/bootstrap.ready"
HF_CACHE_DIR="/opt/falcon-pipeline/hf-cache"
OUTPUT_DIR="/opt/falcon-pipeline/outputs"

mkdir -p "${STATE_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -f "${READY_FILE}" ]]; then
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release python3

if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  cat >/etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable docker
  systemctl start docker
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  systemctl stop google-cloud-ops-agent || true
  mkdir -p /opt/google/cuda-installer
  cd /opt/google/cuda-installer
  if [[ ! -f cuda_installer.pyz ]]; then
    curl -fSsL -o cuda_installer.pyz https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
  fi
  python3 cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod || true
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver installation is still in progress; reboot the VM and rerun validation."
  exit 0
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    >/etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
  apt-get install -y \
    "nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    "nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    "libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    "libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" || \
    apt-get install -y nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

mkdir -p "${HF_CACHE_DIR}" "${OUTPUT_DIR}"
chmod 0777 "${HF_CACHE_DIR}" "${OUTPUT_DIR}"
touch "${READY_FILE}"
