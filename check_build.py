from __future__ import annotations

import sys
from pathlib import Path


def find_dll(build_dir: Path, filename: str) -> list[Path]:
    return [path for path in build_dir.rglob(filename) if path.is_file()]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: check_build.py DIST_DIRECTORY")
        return 1

    build_dir = Path(sys.argv[1]).resolve()
    exe_path = build_dir / "speech_recognition.exe"
    if not exe_path.is_file():
        print(f"ERROR: EXE not found: {exe_path}")
        return 1

    required_assets = (
        build_dir / "_internal" / "assets" / "academy_logo.png",
        build_dir / "_internal" / "assets" / "app_icon.ico",
    )
    missing_assets = [path for path in required_assets if not path.is_file()]
    if missing_assets:
        for path in missing_assets:
            print(f"ERROR: required asset is missing: {path}")
        return 1

    cuda_dlls = {
        "cublas64_12.dll": find_dll(build_dir, "cublas64_12.dll"),
        "cublasLt64_12.dll": find_dll(build_dir, "cublasLt64_12.dll"),
        "cudnn64_9.dll": find_dll(build_dir, "cudnn64_9.dll"),
    }
    missing_cuda = [name for name, paths in cuda_dlls.items() if not paths]

    print(f"OK: {exe_path}")
    if missing_cuda:
        print(
            "WARNING: the following CUDA DLLs were not found: "
            + ", ".join(missing_cuda)
        )
        print(
            "The application will still work on CPU. For a self-contained "
            "CUDA build, place matching CUDA 12/cuDNN 9 DLLs in cuda_dlls "
            "and rebuild."
        )
    else:
        print("OK: CUDA 12/cuDNN 9 runtime DLLs are included.")

    print("OK: if CUDA fails at runtime, the application falls back to CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
