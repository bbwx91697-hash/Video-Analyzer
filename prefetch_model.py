# 预下载 faster-whisper 模型并检查 CUDA 设备可见性。
"""Download a faster-whisper model into the configured Hugging Face cache."""
import argparse
import glob
import os
import sys
from pathlib import Path

# Keep handles alive until process exit: Windows removes a directory when closed.
_DLL_HANDLES = []


# 无输入；Windows 下注册 NVIDIA DLL 目录并保留句柄，无返回值。
def configure_dll_paths():
    if os.name != "nt":
        return
    paths = glob.glob(str(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "*" / "bin"))
    for path in paths:
        _DLL_HANDLES.append(os.add_dll_directory(path))
    if paths:
        os.environ["PATH"] = os.pathsep.join(paths + [os.environ.get("PATH", "")])


# 读取模型参数；报告 CUDA 设备数并下载模型到配置缓存，返回退出码。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()
    configure_dll_paths()
    import ctranslate2
    from faster_whisper.utils import download_model

    try:
        print(f"CUDA devices: {ctranslate2.get_cuda_device_count()}", flush=True)
    except Exception as exc:
        # Runtime/driver availability does not prevent downloading a model for CPU use.
        print(f"CUDA smoke check failed: {type(exc).__name__}", flush=True)
    hf_home = Path(os.environ.get("HF_HOME", str(Path(__file__).resolve().parent / "models" / "hf")))
    # WhisperModel(download_root=HF_HOME) uses this same snapshot cache.
    cache = hf_home
    cache.mkdir(parents=True, exist_ok=True)
    try:
        path = download_model(args.model, cache_dir=str(cache))
    except Exception as exc:
        # Hub errors may contain signed URLs or credentials.
        print(f"Model download failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Model available at: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
