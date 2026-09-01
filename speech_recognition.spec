from pathlib import Path
import os
import site
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


PROJECT_ROOT = Path(SPECPATH).resolve()
APP_NAME = "speech_recognition"

datas = [
    (str(PROJECT_ROOT / "assets"), "assets"),
]
binaries = []
hiddenimports = [
    "faster_whisper.audio",
    "ctranslate2",
    "av",
    "onnxruntime",
    "tokenizers",
    "sounddevice",
    "_sounddevice_data",
    "torch",
    "torchaudio",
    "torchcodec",
    "pyannote.audio",
    "pyannote.audio.pipelines",
    "pyannote.audio.pipelines.speaker_diarization",
    "pyannote.audio.pipelines.clustering",
    "lightning",
    "torchmetrics",
    "speechbrain",
    "wespeaker",
    "omegaconf",
    "safetensors",
    "asteroid_filterbanks",
    "torch_audiomentations",
]


def append_unique(target, entries):
    existing = {tuple(str(value) for value in entry) for entry in target}
    for entry in entries:
        normalized = tuple(str(value) for value in entry)
        if normalized not in existing:
            target.append(entry)
            existing.add(normalized)


# Динамически импортируемые части pyannote и связанных пакетов.
for package_name in (
    "pyannote",
    "lightning",
    "torchmetrics",
    "speechbrain",
    "wespeaker",
    "asteroid_filterbanks",
    "torch_audiomentations",
):
    try:
        hiddenimports.extend(
            collect_submodules(package_name, on_error="warn once")
        )
    except Exception as exc:
        print(f"WARNING: submodules for {package_name} were not collected: {exc}")


# Служебные ресурсы Python-пакетов. Пользовательские модели сюда не входят.
# collect_all(torch) намеренно не используется:
# он копирует огромные деревья лицензий и часто вызывает WinError 206.
for package_name in (
    "faster_whisper",
    "pyannote.audio",
    "speechbrain",
    "wespeaker",
    "omegaconf",
    "safetensors",
    "tokenizers",
    "lightning",
    "torchmetrics",
    "_sounddevice_data",
    "torchcodec",
):
    try:
        append_unique(datas, collect_data_files(package_name))
    except Exception as exc:
        print(f"WARNING: data files for {package_name} were not collected: {exc}")


# Нативные библиотеки CPU, аудиодекодеров, PyTorch и CUDA-сборки PyTorch.
for package_name in (
    "ctranslate2",
    "torch",
    "torchaudio",
    "av",
    "onnxruntime",
    "_sounddevice_data",
    "torchcodec",
):
    try:
        append_unique(binaries, collect_dynamic_libs(package_name))
    except Exception as exc:
        print(f"WARNING: DLL files for {package_name} were not collected: {exc}")


def add_dll_tree(source_root: Path, destination_root: str) -> None:
    if not source_root.is_dir():
        return
    additions = []
    for dll_path in sorted(source_root.rglob("*.dll")):
        relative_parent = dll_path.parent.relative_to(source_root)
        destination = Path(destination_root) / relative_parent
        additions.append((str(dll_path), str(destination)))
    append_unique(binaries, additions)


# 1. DLL, вручную положенные пользователем в cuda_dlls.
add_dll_tree(PROJECT_ROOT / "cuda_dlls", "cuda_dlls")

# 2. NVIDIA runtime, установленный как Python-пакеты (если доступен).
site_package_roots = []
try:
    site_package_roots.extend(Path(path) for path in site.getsitepackages())
except Exception:
    pass
site_package_roots.append(
    Path(sys.prefix) / "Lib" / "site-packages"
)
for package_root in site_package_roots:
    add_dll_tree(package_root / "nvidia", "nvidia")

# 3. CUDA Toolkit/cuDNN, установленные в Windows.
cuda_patterns = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "cudnn*.dll",
    "nvrtc64_*.dll",
    "nvrtc-builtins64_*.dll",
    "zlibwapi.dll",
)
for environment_name in ("CUDA_PATH", "CUDNN_PATH"):
    environment_value = os.environ.get(environment_name)
    if not environment_value:
        continue
    bin_dir = Path(environment_value) / "bin"
    if not bin_dir.is_dir():
        continue
    additions = []
    for pattern in cuda_patterns:
        additions.extend((str(path), "cuda_dlls") for path in bin_dir.glob(pattern))
    append_unique(binaries, additions)


hiddenimports = sorted(set(hiddenimports))

a = Analysis(
    [str(PROJECT_ROOT / "app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(PROJECT_ROOT / "assets" / "app_icon.ico"),
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
