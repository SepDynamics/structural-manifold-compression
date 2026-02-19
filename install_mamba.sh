#!/bin/bash
# Install mamba-ssm with proper GCC compiler for RTX 3080 Ti ==="
echo ""

set -euo pipefail

# Get script directory for absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

# Auto-detect compatible GCC for CUDA compilation
if command -v gcc-11 &> /dev/null; then
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    echo "✓ Using GCC 11 for CUDA compatibility"
else
    echo "✗ Error: No compatible GCC found (gcc-14 or gcc-11 required)"
    exit 1
fi
export CUDAHOSTCXX="${CXX}"

# Set optimal CUDA arch for RTX 3080 Ti (Ampere architecture) - only 8.6 to speed up build
export TORCH_CUDA_ARCH_LIST="8.6"

# Fix for glibc/CUDA cospi/sinpi conflict (Glibc 2.38+ vs CUDA 12) and missing std::min/max in device headers
COMMON_HOST_PREFIX="${SCRIPT_DIR}/cmake/cuda_host_prefix.h"
export COMMON_MATH_GUARDS="-D__MATH_FUNCTIONS_H__ -D__CUDA_MATH_FUNCTIONS_H__ -D__CUDA_MATH_FUNCTIONS_HPP__"
export COMMON_HOST_INCLUDES="-include algorithm -include cmath -include ${COMMON_HOST_PREFIX}"
export CFLAGS="${COMMON_MATH_GUARDS} ${COMMON_HOST_INCLUDES} -Wno-error ${CFLAGS-}"
export CXXFLAGS="${COMMON_MATH_GUARDS} ${COMMON_HOST_INCLUDES} -Wno-error ${CXXFLAGS-}"
NVCC_COMMON_FLAGS="${COMMON_MATH_GUARDS} -include ${COMMON_HOST_PREFIX} --extended-lambda -Wno-deprecated-gpu-targets"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS-} ${NVCC_COMMON_FLAGS} -Xcompiler=-include -Xcompiler=${COMMON_HOST_PREFIX}"
export TORCH_NVCC_FLAGS="${TORCH_NVCC_FLAGS-} ${NVCC_COMMON_FLAGS}"

fetch_sdist() {
    local package_name="$1"
    local version="$2"
    local dest="$3"
    python3 - <<'PY' "${package_name}" "${version}" "${dest}"
import json
import pathlib
import sys
import urllib.request
pkg, ver, dest = sys.argv[1:4]
url = f"https://pypi.org/pypi/{pkg}/{ver}/json"
with urllib.request.urlopen(url) as resp:
    data = json.load(resp)
for file in data["urls"]:
    if file["packagetype"] == "sdist":
        download_url = file["url"]
        break
else:
    raise SystemExit(f"No sdist found for {pkg}=={ver}")
path = pathlib.Path(dest)
path.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(download_url) as resp, open(path, "wb") as fh:
    fh.write(resp.read())
print(download_url)
PY
}

add_host_prefix_include() {
    local file="$1"
    if ! grep -q "cuda_host_prefix.h" "${file}"; then
        printf '#include "%s"\n\n' "${COMMON_HOST_PREFIX}" | cat - "${file}" > "${file}.patched" && mv "${file}.patched" "${file}"
    fi
}

patch_package_sources() {
    local pkg_dir="$1"
    shift
    local patterns=("$@")
    for pattern in "${patterns[@]}"; do
        while IFS= read -r -d '' file; do
            add_host_prefix_include "${file}"
        done < <(find "${pkg_dir}" -type f -name "${pattern}" -print0)
    done
}

install_from_source() {
    local package_name="$1"
    local version="$2"
    shift 2
    local patterns=("$@")
    local tmp_dir tarball src_dir
    tmp_dir="$(mktemp -d)"
    tarball="${tmp_dir}/${package_name}-${version}.tar.gz"
    echo "→ Downloading ${package_name}==${version} source..."
    fetch_sdist "${package_name}" "${version}" "${tarball}" >/dev/null
    tar -xf "${tarball}" -C "${tmp_dir}"
    src_dir=$(find "${tmp_dir}" -maxdepth 1 -type d -name "${package_name}-*" | head -n 1)
    patch_package_sources "${src_dir}" "${patterns[@]}"
    echo "→ Installing patched ${package_name}..."
    python3 -m pip install --no-build-isolation --no-deps "${src_dir}"
    rm -rf "${tmp_dir}"
}

echo ""
echo "Installing patched mamba-ssm and causal-conv1d..."
echo "This may take 5-10 minutes as it compiles CUDA kernels for RTX 3080 Ti..."
echo ""

install_from_source "causal-conv1d" "1.6.0" "*.cu"
install_from_source "mamba-ssm" "2.3.0" "*.cu" "*.cuh"

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "Test the installation with:"
echo "  python3 -c 'import mamba_ssm; print(\"✓ mamba-ssm installed successfully\")'"
echo ""
