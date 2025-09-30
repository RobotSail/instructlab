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

# Step 1: Fix kernel package consistency for RUNNING kernel
log_info "Step 1: Ensuring kernel packages match RUNNING kernel"

# CRITICAL: We must match the RUNNING kernel, not just any installed kernel
RUNNING_KERNEL=$(uname -r)
RUNNING_KERNEL_VERSION=$(echo ${RUNNING_KERNEL} | cut -d'-' -f1-2)

log_info "  Running kernel: ${RUNNING_KERNEL}"

# Check if we have matching kernel-devel and kernel-headers for the running kernel
RUNNING_KERNEL_DEVEL="kernel-devel-${RUNNING_KERNEL_VERSION}"
RUNNING_KERNEL_HEADERS="kernel-headers-${RUNNING_KERNEL_VERSION}"

# Install matching kernel-devel if not present
if ! rpm -q "${RUNNING_KERNEL_DEVEL}" &>/dev/null; then
    log_info "Installing kernel-devel for running kernel: ${RUNNING_KERNEL_VERSION}"
    dnf install -y "${RUNNING_KERNEL_DEVEL}"
else
    log_info "  kernel-devel: ${RUNNING_KERNEL_VERSION} (OK)"
fi

# Install matching kernel-headers if not present or mismatched
if ! rpm -q "${RUNNING_KERNEL_HEADERS}" &>/dev/null; then
    log_info "Installing kernel-headers for running kernel: ${RUNNING_KERNEL_VERSION}"
    dnf install -y "${RUNNING_KERNEL_HEADERS}"
else
    log_info "  kernel-headers: ${RUNNING_KERNEL_VERSION} (OK)"
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
    elfutils-libelf-devel \
    nvtop

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

# Step 6c2: Install cuDNN and CUDA compatibility libraries
log_info "Step 6c2: Installing cuDNN for deep learning and CUDA compatibility"
dnf install -y \
    cudnn9-cuda-${CUDA_DASHED_VERSION} \
    libcudnn8 \
    cuda-compat-${CUDA_DASHED_VERSION}

# Determine NCCL package version and install if available
NCCL_PACKAGE=$(dnf list available --showduplicates 2>/dev/null | \
    grep "libnccl-2" | grep "${CUDA_MAJOR_MINOR}" | tail -1 | awk '{print $1}' || echo "")

if [[ -n "${NCCL_PACKAGE}" ]]; then
    log_info "Installing NCCL: ${NCCL_PACKAGE}"
    dnf install -y "${NCCL_PACKAGE}"
else
    log_warn "Could not find NCCL package for CUDA ${CUDA_MAJOR_MINOR}"
fi

# Step 6c3: Install Fabric Manager for multi-GPU NVLink/NVSwitch
log_info "Step 6c3: Installing Fabric Manager for multi-GPU systems"

# Calculate driver branch from driver version
DRIVER_BRANCH=$(echo ${DRIVER_VERSION} | cut -d'.' -f1)

log_info "  Driver branch: ${DRIVER_BRANCH}"
log_info "  Installing nvidia-fabric-manager-${DRIVER_VERSION}"

dnf install -y \
    nvidia-fabric-manager-${DRIVER_VERSION} \
    libnvidia-nscq-${DRIVER_BRANCH}-${DRIVER_VERSION}

# Step 6d: Build DKMS modules for all installed kernels
log_info "Step 6d: Building DKMS modules for all installed kernels"

# Get NVIDIA driver version from DKMS
NVIDIA_DKMS_VERSION=$(dkms status nvidia-open 2>/dev/null | head -1 | cut -d',' -f1 | cut -d'/' -f2 || echo "")

if [[ -z "${NVIDIA_DKMS_VERSION}" ]]; then
    log_warn "Could not determine NVIDIA DKMS version, attempting to use installed driver version"
    NVIDIA_DKMS_VERSION=$(rpm -q nvidia-driver --queryformat '%{VERSION}' 2>/dev/null)
fi

if [[ -n "${NVIDIA_DKMS_VERSION}" ]]; then
    log_info "  NVIDIA DKMS version: ${NVIDIA_DKMS_VERSION}"

    # Build for all installed kernels
    for KVER in $(ls /lib/modules/); do
        if [[ -d "/lib/modules/${KVER}/build" ]]; then
            log_info "  Building for kernel: ${KVER}"
            dkms install "nvidia-open/${NVIDIA_DKMS_VERSION}" -k "${KVER}" 2>&1 | grep -E "(Building|Installing|already)" || true
        fi
    done
else
    log_error "Could not determine NVIDIA DKMS version"
fi

# Step 6e: Configure CUDA environment variables
log_info "Step 6e: Configuring CUDA environment variables"

cat > /etc/profile.d/cuda.sh <<'EOF'
#!/bin/bash
# CUDA environment configuration
# Generated by NVIDIA installation script

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
EOF

chmod +x /etc/profile.d/cuda.sh
log_info "  Created /etc/profile.d/cuda.sh"

# Step 6f: Load NVIDIA kernel modules
log_info "Step 6f: Loading NVIDIA kernel modules"

if modprobe nvidia-drm 2>/dev/null; then
    log_info "  NVIDIA kernel modules loaded successfully"

    # Verify with nvidia-smi
    if nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        log_info "  Detected ${GPU_COUNT} GPU(s)"
    else
        log_warn "  nvidia-smi failed - may need reboot"
    fi
else
    log_warn "  Could not load NVIDIA modules - reboot required"
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
systemctl enable nvidia-fabricmanager.service

# Start fabric manager if NVIDIA modules are loaded
if lsmod | grep -q nvidia; then
    log_info "Starting Fabric Manager service"
    systemctl start nvidia-fabricmanager.service

    # Verify it started successfully
    if systemctl is-active --quiet nvidia-fabricmanager.service; then
        log_info "  Fabric Manager started successfully"
    else
        log_warn "  Fabric Manager failed to start - check logs: journalctl -u nvidia-fabricmanager"
    fi
else
    log_info "  Fabric Manager will start after reboot when NVIDIA modules load"
fi

# Blacklist nouveau
echo "blacklist nouveau" > /etc/modprobe.d/blacklist_nouveau.conf

# Step 8: Lock versions for reproducibility
log_info "Step 8: Locking package versions for reproducibility"

# Lock all NVIDIA packages
dnf versionlock delete 'nvidia-*' 'cuda-*' 'libnccl*' 'libcudnn*' 'kmod-nvidia*' 2>/dev/null || true

dnf versionlock add \
    nvidia-driver \
    nvidia-driver-cuda \
    nvidia-driver-libs \
    nvidia-driver-NVML \
    nvidia-persistenced \
    nvidia-settings \
    kmod-nvidia-open-dkms \
    nvidia-fabric-manager \
    libnvidia-nscq-${DRIVER_BRANCH} \
    cuda-toolkit-${CUDA_DASHED_VERSION} \
    cudnn9-cuda-${CUDA_DASHED_VERSION} \
    libcudnn8 \
    cuda-compat-${CUDA_DASHED_VERSION} \
    nvidia-container-toolkit

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

# DKMS Status:
$(dkms status 2>/dev/null || echo "DKMS not available")

# Loaded Kernel Modules:
$(lsmod | grep nvidia || echo "No NVIDIA modules loaded")

# GPU Detection:
$(nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available or no GPUs detected")

# Installed Packages:
EOF

dnf list installed | grep -E "(nvidia|cuda)" >> "${MANIFEST_FILE}" || true

cat >> "${MANIFEST_FILE}" <<EOF

# Version Locks:
$(dnf versionlock list 2>/dev/null | grep -E "(nvidia|cuda|kmod)" || echo "No version locks found")
EOF

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

# Check if modules are loaded and GPUs are detected
if nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_info "✓ NVIDIA driver is active and ${GPU_COUNT} GPU(s) detected"
    log_info ""
    log_info "Verify installation with:"
    log_info "  nvidia-smi"
    log_info "  source /etc/profile.d/cuda.sh && nvcc --version"
    log_info ""
    log_info "Note: For nvcc to work in new shells, users must either:"
    log_info "  - Log out and log back in, OR"
    log_info "  - Run: source /etc/profile.d/cuda.sh"
else
    log_warn "NVIDIA driver is installed but not loaded."
    log_warn "You must reboot the system for the driver to load."
    log_info ""
    log_info "After reboot, verify installation with:"
    log_info "  nvidia-smi"
    log_info "  nvcc --version"
fi

log_info ""
log_info "To replicate this installation on another node:"
log_info "  1. Copy this script to the new node"
log_info "  2. Run: bash $(basename $0)"
log_info "  3. The driver will load automatically (or reboot if needed)"
log_info ""
