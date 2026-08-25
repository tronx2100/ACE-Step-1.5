"""Finds the venv's own pip-installed CUDA (nvidia-*-cu12) library directories.

On hosts where LD_LIBRARY_PATH already points at a system CUDA toolkit
install (e.g. /usr/local/cuda/lib64) ahead of the venv, dynamic libraries
resolve to a mix of versions - e.g. libcublasLt from the system CUDA 12.9
toolkit alongside libcublas from the pip-installed nvidia-cublas-cu12
12.8.4 wheel. That ABI mismatch reliably crashes CUDA calls (PyTorch and
ctranslate2 alike) with CUBLAS_STATUS_INVALID_VALUE or a CUDA
initialization error. Putting the venv's own matched set of libraries
first on LD_LIBRARY_PATH fixes it.
"""

import os
import sys


def venv_cuda_lib_dirs() -> list[str]:
    site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    nvidia_root = None
    if os.path.isdir(site_packages):
        for entry in os.listdir(site_packages):
            candidate = os.path.join(site_packages, entry, "site-packages", "nvidia")
            if os.path.isdir(candidate):
                nvidia_root = candidate
                break
    if nvidia_root is None:
        return []
    dirs = []
    for pkg in ("cublas", "cudnn", "cuda_runtime", "cusparse", "cufft", "curand", "cusolver", "nvjitlink"):
        lib_dir = os.path.join(nvidia_root, pkg, "lib")
        if os.path.isdir(lib_dir):
            dirs.append(lib_dir)
    return dirs
