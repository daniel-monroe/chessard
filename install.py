#!/usr/bin/env python3
"""Install chessard on Linux, macOS or Windows: from a fresh clone to a working UCI engine.

    python install.py              # Linux / macOS (python3 install.py if python is Python 2)
    py install.py                  # Windows

Uses only the standard library, so any Python 3.10+ can run it. Re-running only redoes what
is missing. Everything goes inside this folder:

    venv/        Python environment, with the engine command venv/bin/chessard
                 (venv\\Scripts\\chessard.exe on Windows)
    stockfish/   Stockfish (the official release for this OS; built from source on Linux
                 when the release binary can't run, e.g. on an old glibc)
    weights/     chessard.pt + loras/, from huggingface.co/danielgmonroe/chessard

Options:
    --cpu / --cuda       which PyTorch build (default: CUDA if nvidia-smi works, on Linux/Windows)
    --weights-src DIR    copy the weights from a local folder instead of downloading them
    --skip-verify        don't run verify.py at the end
"""
import argparse
import hashlib
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import venv
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, "venv")
WEIGHTS = os.path.join(HERE, "weights")
SF_DIR = os.path.join(HERE, "stockfish")

TORCH = "2.14.1"
TORCH_INTEL_MAC = "2.2.2"           # PyTorch stopped publishing Intel macOS builds after 2.2
SF_TAG = "sf_18"
HF_REPO = "danielgmonroe/chessard"

WINDOWS = os.name == "nt"
SYSTEM = platform.system()          # Linux / Darwin / Windows
MACHINE = platform.machine().lower()
ARM = MACHINE in ("arm64", "aarch64")
INTEL_MAC = SYSTEM == "Darwin" and not ARM
EXE = ".exe" if WINDOWS else ""
VPY = os.path.join(VENV, "Scripts" if WINDOWS else "bin", "python" + EXE)
ENGINE = os.path.join(VENV, "Scripts" if WINDOWS else "bin", "chessard" + EXE)
SF_BIN = os.path.join(SF_DIR, "stockfish" + EXE)


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def die(msg: str) -> None:
    sys.exit(f"\nerror: {msg}")


def run(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def pip(*args: str) -> None:
    run(VPY, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *args)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "chessard-install"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ------------------------------------------------------------------------------- python
def make_venv() -> None:
    step("1/4 Python environment -> venv/")
    if sys.version_info < (3, 10):
        die(f"chessard needs Python 3.10 or newer (this is {platform.python_version()})")
    if INTEL_MAC and sys.version_info >= (3, 13):
        die("on an Intel Mac, PyTorch only goes up to Python 3.12; run this with python3.12")
    if not os.path.isfile(VPY):
        try:
            # symlinks as `python -m venv` does on macOS/Linux: a copied interpreter from a
            # uv/pyenv-style install can't find its standard library.
            venv.create(VENV, with_pip=True, symlinks=not WINDOWS)
        except Exception as e:
            die(f"could not create a virtualenv ({e}).\n"
                "       On Debian/Ubuntu: sudo apt install python3-venv")
    pip("--upgrade", "pip")


def torch_install_args(flavor: str) -> tuple[str, list[str]]:
    """(expected torch.__version__ prefix, pip arguments) for this machine."""
    if SYSTEM == "Darwin":
        v = TORCH_INTEL_MAC if INTEL_MAC else TORCH
        return v, [f"torch=={v}"]
    if flavor == "cuda":
        if WINDOWS:   # PyPI only has the CPU build for Windows
            return f"{TORCH}+cu126", [f"torch=={TORCH}+cu126",
                                       "--index-url", "https://download.pytorch.org/whl/cu126"]
        return TORCH, [f"torch=={TORCH}"]           # Linux: PyPI's default build is CUDA
    if SYSTEM == "Linux" and not ARM:               # PyPI's x86 Linux build would pull in CUDA
        return f"{TORCH}+cpu", [f"torch=={TORCH}+cpu",
                                "--index-url", "https://download.pytorch.org/whl/cpu"]
    return TORCH, [f"torch=={TORCH}"]


def install_packages(flavor: str) -> None:
    want, args = torch_install_args(flavor)
    have = subprocess.run([VPY, "-c", "import torch; print(torch.__version__)"],
                          capture_output=True, text=True).stdout.strip()
    cuda_build = "+cu" in have or (SYSTEM == "Linux" and "+" not in have and have != "")
    flavor_ok = (flavor == "cuda") == cuda_build or SYSTEM == "Darwin"
    if have.startswith(want) and flavor_ok:
        info(f"torch {have} already installed")
    else:
        if have:
            info(f"replacing torch {have}")
            run(VPY, "-m", "pip", "uninstall", "--quiet", "-y", "torch")
        size = "~2.5 GB" if flavor == "cuda" else "~200 MB"
        info(f"installing torch {want} ({flavor if SYSTEM != 'Darwin' else 'macOS'} build, {size})")
        pip(*args)
    info("installing requirements.txt and the chessard command")
    pip("-r", os.path.join(HERE, "requirements.txt"))
    pip("--no-deps", "-e", HERE)
    run(VPY, "-c", "import chess, numpy, torch; print(f'    torch {torch.__version__} "
                   "(CUDA: {torch.cuda.is_available()}), chess {chess.__version__}, "
                   "numpy {numpy.__version__}')")


# ---------------------------------------------------------------------------- stockfish
def sf_works(path: str) -> bool:
    try:
        r = subprocess.run([path], input="uci\nquit\n", capture_output=True, text=True, timeout=30)
        return "uciok" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def sf_release_assets() -> list[str]:
    """Official sf_18 release builds for this machine, fastest first. Each is test-run before it
    is used, so one this CPU can't execute (missing AVX2, old glibc, ...) is just skipped."""
    if WINDOWS:
        return (["windows-armv8-dotprod.zip", "windows-armv8.zip"] if ARM else
                ["windows-x86-64-avx2.zip", "windows-x86-64-sse41-popcnt.zip", "windows-x86-64.zip"])
    if SYSTEM == "Darwin":
        return (["macos-m1-apple-silicon.tar"] if ARM else
                ["macos-x86-64-avx2.tar", "macos-x86-64-sse41-popcnt.tar", "macos-x86-64.tar"])
    if SYSTEM == "Linux" and not ARM:
        return ["ubuntu-x86-64-avx2.tar", "ubuntu-x86-64-sse41-popcnt.tar", "ubuntu-x86-64.tar"]
    return []   # Linux ARM: no official build, compile it


def extract_binary(data: bytes, name: str, dest: str) -> str | None:
    """Pull the stockfish executable out of a release archive into dest/; return its path."""
    if name.endswith(".zip"):
        arc = zipfile.ZipFile(io.BytesIO(data))
        members = [(m.filename, m.file_size) for m in arc.infolist() if not m.is_dir()]
        read = lambda n: arc.read(n)
    else:
        arc = tarfile.open(fileobj=io.BytesIO(data))
        members = [(m.name, m.size) for m in arc.getmembers() if m.isfile()]
        read = lambda n: arc.extractfile(n).read()
    exes = [(size, n) for n, size in members
            if os.path.basename(n).startswith("stockfish") and (n.endswith(".exe") or "." not in os.path.basename(n))]
    if not exes:
        return None
    out = os.path.join(dest, "stockfish" + EXE)
    with open(out, "wb") as f:
        f.write(read(max(exes)[1]))      # the binary is by far the largest match
    os.chmod(out, 0o755)
    return out


def build_stockfish(tmp: str) -> str:
    make = shutil.which("make")
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not (make and cxx):
        die("no Stockfish release runs here, and building it needs make and a C++ compiler\n"
            "       (Debian/Ubuntu: sudo apt install build-essential)")
    info(f"downloading the Stockfish {SF_TAG} source")
    src_tar = tarfile.open(fileobj=io.BytesIO(fetch(
        f"https://github.com/official-stockfish/Stockfish/archive/refs/tags/{SF_TAG}.tar.gz")))
    if hasattr(tarfile, "data_filter"):     # Python 3.12+: refuse unsafe paths in the archive
        src_tar.extractall(tmp, filter="data")
    else:
        src_tar.extractall(tmp)
    src = os.path.join(tmp, f"Stockfish-{SF_TAG}", "src")
    flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
    arch = "armv8" if ARM else ("x86-64-avx2" if " avx2" in flags else "x86-64")
    info(f"building it (ARCH={arch}); this takes a few minutes")
    run(make, f"-j{os.cpu_count() or 2}", "build", f"ARCH={arch}", cwd=src, stdout=subprocess.DEVNULL)
    return os.path.join(src, "stockfish")


def install_stockfish() -> None:
    step(f"2/4 Stockfish ({SF_TAG}) -> stockfish/")
    if sf_works(SF_BIN):
        info("already installed")
        return
    os.makedirs(SF_DIR, exist_ok=True)
    base = f"https://github.com/official-stockfish/Stockfish/releases/download/{SF_TAG}"
    for asset in sf_release_assets():
        info(f"trying stockfish-{asset}")
        try:
            path = extract_binary(fetch(f"{base}/stockfish-{asset}"), asset, SF_DIR)
        except Exception as e:
            info(f"  download failed: {e}")
            continue
        if path and sf_works(path):
            info(f"installed stockfish-{asset}")
            return
        info("  doesn't run on this machine, trying the next build")
    with tempfile.TemporaryDirectory() as tmp:
        built = build_stockfish(tmp)
        shutil.copy2(built, SF_BIN)
    if not sf_works(SF_BIN):
        die("the Stockfish build does not run")
    info("built and installed from source")


# ------------------------------------------------------------------------------ weights
def checksums() -> dict[str, str]:
    out = {}
    with open(os.path.join(HERE, "weights.sha256")) as f:
        for line in f:
            if line.strip():
                digest, name = line.split(None, 1)
                out[name.strip().lstrip("*")] = digest
    return out


def weights_ok(sums: dict[str, str], verbose: bool = False) -> bool:
    ok = True
    for name, digest in sums.items():
        p = os.path.join(WEIGHTS, name)
        h = hashlib.sha256()
        if os.path.isfile(p):
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        good = os.path.isfile(p) and h.hexdigest() == digest
        if verbose and not good:
            info(f"  {'missing' if not os.path.isfile(p) else 'checksum mismatch'}: {name}")
        ok &= good
    return ok


def install_weights(src: str | None) -> None:
    step("3/4 weights -> weights/")
    sums = checksums()
    if weights_ok(sums):
        info("all files present, checksums match")
        return
    if src:
        info(f"copying from {src}")
        for name in sums:
            os.makedirs(os.path.dirname(os.path.join(WEIGHTS, name)), exist_ok=True)
            shutil.copy2(os.path.join(src, name), os.path.join(WEIGHTS, name))
    else:
        info(f"downloading {HF_REPO} from Hugging Face (~400 MB, public, no login needed)")
        try:
            run(VPY, "-c", "import sys; from huggingface_hub import snapshot_download; "
                           "snapshot_download(sys.argv[1], local_dir=sys.argv[2])", HF_REPO, WEIGHTS)
        except subprocess.CalledProcessError:
            die("the download failed (network or proxy?). Re-run to resume, or pass --weights-src DIR")
    if not weights_ok(sums, verbose=True):
        die("the weights don't match weights.sha256 (corrupt download?). Delete weights/ and re-run.")
    info(f"verified {len(sums)} files against weights.sha256")


# --------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cpu", action="store_true", help="CPU-only PyTorch (~200 MB instead of ~2.5 GB)")
    g.add_argument("--cuda", action="store_true", help="CUDA PyTorch (Linux/Windows with an NVIDIA GPU)")
    ap.add_argument("--weights-src", default=os.environ.get("CHESSARD_WEIGHTS_SRC"),
                    help="copy the weights from this folder instead of downloading them")
    ap.add_argument("--skip-verify", action="store_true", help="don't run verify.py at the end")
    a = ap.parse_args()

    if a.cuda:
        flavor = "cuda"
    elif a.cpu or SYSTEM == "Darwin":
        flavor = "cpu"
    else:
        nv = shutil.which("nvidia-smi")
        flavor = "cuda" if nv and subprocess.run([nv, "-L"], capture_output=True).returncode == 0 else "cpu"
    print(f"chessard install: {SYSTEM} {MACHINE}, Python {platform.python_version()}, torch build: "
          f"{'macOS' if SYSTEM == 'Darwin' else flavor}")

    try:
        make_venv()
        install_packages(flavor)
        install_stockfish()
        install_weights(a.weights_src)
    except subprocess.CalledProcessError as e:
        die(f"this command failed (exit {e.returncode}); see its output above:\n       "
            + " ".join(map(str, e.cmd)))

    step("4/4 done")
    rel = os.path.relpath(ENGINE, os.getcwd())
    info(f"engine command: {ENGINE}")
    print(f"""
    Run it:             {rel}                 (Elo 2200; --elo 2000-2900)
    Play as a player:   {rel} --player carlsen
    In a chess GUI:     add a UCI engine with the path {ENGINE}
""", flush=True)
    if not a.skip_verify:
        sys.exit(subprocess.run([sys.executable, os.path.join(HERE, "verify.py")]).returncode)


if __name__ == "__main__":
    main()
