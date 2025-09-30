#!/bin/bash
# Reproducible NVIDIA Driver and CUDA Installation for CentOS Stream 9
# Optimized for H100 GPUs using pre-built RPM packages
#
# This script installs:
# - CUDA Toolkit 12.4.1
# - NVIDIA Driver 550.x (open kernel modules)
# - All dependencies via official NVIDIA repository
#
# Version locking ensures reproducibility across nodes

set -euo pipefail

# Configuration - Update these versions as needed
CUDA_VERSION="12.4.1"
CUDA_MAJOR_MINOR="12.4"
CUDA_DASHED_VERSION="12-4"
DRIVER_STREAM="550"
OS_VERSION_MAJOR="9"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check root privileges
if [[ $(id -u) != "0" ]]; then
    log_error "You must run this script as root."
    exit 1
fi

log_info "Starting NVIDIA driver and CUDA installation for CentOS Stream 9"

# Detect system configuration
KERNEL_VERSION=$(uname -r)
BUILD_ARCH=$(arch)
TARGET_ARCH=${BUILD_ARCH}

log_info "System configuration:"
log_info "  Kernel: ${KERNEL_VERSION}"
log_info "  Architecture: ${BUILD_ARCH}"
log_info "  CUDA Version: ${CUDA_VERSION}"
log_info "  Driver Stream: ${DRIVER_STREAM}"

# Step 1: Fix kernel-headers mismatch if needed
log_info "Step 1: Checking kernel package consistency"

KERNEL_CORE_VERSION=$(rpm -q kernel-core --queryformat '%{VERSION}-%{RELEASE}')
KERNEL_DEVEL_VERSION=$(rpm -q kernel-devel --queryformat '%{VERSION}-%{RELEASE}' 2>/dev/null || echo "not-installed")
KERNEL_HEADERS_VERSION=$(rpm -q kernel-headers --queryformat '%{VERSION}-%{RELEASE}' 2>/dev/null || echo "not-installed")

log_info "  kernel-core: ${KERNEL_CORE_VERSION}"
log_info "  kernel-devel: ${KERNEL_DEVEL_VERSION}"
log_info "  kernel-headers: ${KERNEL_HEADERS_VERSION}"

if [[ "${KERNEL_CORE_VERSION}" != "${KERNEL_HEADERS_VERSION}" ]]; then
    log_warn "Kernel headers mismatch detected. Fixing..."
    dnf downgrade -y "kernel-headers-${KERNEL_CORE_VERSION}" || \
    dnf install -y "kernel-headers-${KERNEL_CORE_VERSION}"
fi

if [[ "${KERNEL_DEVEL_VERSION}" == "not-installed" ]] || [[ "${KERNEL_CORE_VERSION}" != "${KERNEL_DEVEL_VERSION}" ]]; then
    log_info "Installing matching kernel-devel package"
    dnf install -y "kernel-devel-${KERNEL_CORE_VERSION}"
fi

# Step 2: Install prerequisites
log_info "Step 2: Installing prerequisites"

# Install EPEL if not already present
if ! rpm -q epel-release &>/dev/null; then
    dnf install -y epel-release || \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-${OS_VERSION_MAJOR}.noarch.rpm
else
    log_info "EPEL already installed"
fi

dnf install -y 'dnf-command(versionlock)' \
    pciutils \
    gcc \
    make \
    dkms \
    kernel-devel \
    kernel-headers \
    elfutils-libelf-devel

# Step 3: Add NVIDIA CUDA repository
log_info "Step 3: Configuring NVIDIA CUDA repository"

CUDA_REPO_ARCH=${TARGET_ARCH}
if [[ "${TARGET_ARCH}" == "aarch64" ]]; then
    CUDA_REPO_ARCH="sbsa"
fi

CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION_MAJOR}/${CUDA_REPO_ARCH}/cuda-rhel${OS_VERSION_MAJOR}.repo"

if [[ ! -f /etc/yum.repos.d/cuda-rhel${OS_VERSION_MAJOR}.repo ]]; then
    dnf config-manager --add-repo "${CUDA_REPO_URL}"
    log_info "Added CUDA repository"
else
    log_info "CUDA repository already configured"
fi

# Step 4: Enable NVIDIA driver module stream (open kernel modules for H100)
log_info "Step 4: Enabling NVIDIA driver module stream: ${DRIVER_STREAM}-open"
dnf module reset -y nvidia-driver 2>/dev/null || true
dnf module enable -y "nvidia-driver:${DRIVER_STREAM}-open"

# Step 5: Determine exact driver version available
log_info "Step 5: Determining available driver version"
dnf makecache

# Get the latest available driver version in the 550 stream
DRIVER_VERSION=$(dnf list available nvidia-driver --showduplicates 2>/dev/null | \
    grep "^nvidia-driver" | grep "550" | tail -1 | awk '{print $2}' | cut -d':' -f2 | cut -d'-' -f1)

if [[ -z "${DRIVER_VERSION}" ]]; then
    log_error "Could not determine driver version from repository"
    exit 1
fi

log_info "  Selected driver version: ${DRIVER_VERSION}"

# Step 6: Install NVIDIA drivers via module
log_info "Step 6: Installing NVIDIA drivers via module"
dnf module install -y nvidia-driver:${DRIVER_STREAM}-open/default

# Determine exact driver version that was installed
DRIVER_VERSION=$(rpm -q nvidia-driver --queryformat '%{VERSION}' 2>/dev/null || echo "unknown")
log_info "  Installed driver version: ${DRIVER_VERSION}"

# Step 6b: Install CUDA toolkit
log_info "Step 6b: Installing CUDA toolkit ${CUDA_VERSION}"
dnf install -y cuda-toolkit-${CUDA_DASHED_VERSION}

# Step 6c: Install NVIDIA Container Toolkit
log_info "Step 6c: Installing NVIDIA Container Toolkit"
dnf install -y nvidia-container-toolkit

# Determine NCCL package version and install if available
NCCL_PACKAGE=$(dnf list available --showduplicates 2>/dev/null | \
    grep "libnccl-2" | grep "${CUDA_MAJOR_MINOR}" | tail -1 | awk '{print $1}' || echo "")

if [[ -n "${NCCL_PACKAGE}" ]]; then
    log_info "Installing NCCL: ${NCCL_PACKAGE}"
    dnf install -y "${NCCL_PACKAGE}"
else
    log_warn "Could not find NCCL package for CUDA ${CUDA_MAJOR_MINOR}"
fi

# Step 7: Configure NVIDIA services
log_info "Step 7: Configuring NVIDIA services"

# Create nvidia-toolkit-setup service if not exists
mkdir -p /usr/lib/systemd/system

if [[ ! -f /usr/lib/systemd/system/nvidia-toolkit-setup.service ]]; then
    cat > /usr/lib/systemd/system/nvidia-toolkit-setup.service <<'EOF'
[Unit]
Description=Generate /etc/cdi/nvidia.yaml
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
fi

# Enable services
systemctl daemon-reload
systemctl enable nvidia-persistenced.service
systemctl enable nvidia-toolkit-setup.service

# Blacklist nouveau
echo "blacklist nouveau" > /etc/modprobe.d/blacklist_nouveau.conf

# Step 8: Lock versions for reproducibility
log_info "Step 8: Locking package versions for reproducibility"

# Lock all NVIDIA packages
dnf versionlock delete 'nvidia-*' 'cuda-*' 'libnccl*' 'libcudnn*' 2>/dev/null || true

dnf versionlock add \
    nvidia-open \
    nvidia-driver-cuda-${DRIVER_VERSION} \
    nvidia-driver-libs-${DRIVER_VERSION} \
    nvidia-driver-NVML-${DRIVER_VERSION} \
    nvidia-persistenced-${DRIVER_VERSION} \
    cuda-toolkit-${CUDA_DASHED_VERSION} \
    cuda-drivers-${DRIVER_STREAM}

if [[ -n "${NCCL_PACKAGE}" ]]; then
    dnf versionlock add "${NCCL_PACKAGE}"
fi

# Step 9: Export package manifest for documentation
log_info "Step 9: Creating package manifest"

MANIFEST_FILE="/root/nvidia-install-manifest-$(date +%Y%m%d-%H%M%S).txt"

cat > "${MANIFEST_FILE}" <<EOF
# NVIDIA Driver and CUDA Installation Manifest
# Generated: $(date)
# System: CentOS Stream ${OS_VERSION_MAJOR}
# Kernel: ${KERNEL_VERSION}
# Architecture: ${BUILD_ARCH}

# Configuration
CUDA_VERSION=${CUDA_VERSION}
DRIVER_VERSION=${DRIVER_VERSION}
DRIVER_STREAM=${DRIVER_STREAM}

# Installed Packages:
EOF

dnf list installed | grep -E "(nvidia|cuda)" >> "${MANIFEST_FILE}" || true

log_info "Package manifest saved to: ${MANIFEST_FILE}"

# Step 10: Show versionlock status
log_info "Step 10: Locked packages:"
dnf versionlock list | grep -E "(nvidia|cuda)" || log_warn "No packages locked"

# Summary
echo ""
log_info "========================================"
log_info "Installation completed successfully!"
log_info "========================================"
log_info ""
log_info "Installed versions:"
log_info "  CUDA Toolkit: ${CUDA_VERSION}"
log_info "  NVIDIA Driver: ${DRIVER_VERSION}"
log_info "  Driver Type: Open Kernel Modules"
log_info ""
log_info "Package manifest: ${MANIFEST_FILE}"
log_info ""
log_warn "IMPORTANT: You must reboot the system for the driver to load."
log_info ""
log_info "After reboot, verify installation with:"
log_info "  nvidia-smi"
log_info "  nvcc --version"
log_info ""
log_info "To replicate this installation on another node:"
log_info "  1. Copy this script to the new node"
log_info "  2. Run: bash $(basename $0)"
log_info "  3. Reboot"
log_info ""
