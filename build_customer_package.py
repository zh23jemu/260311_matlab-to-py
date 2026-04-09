from __future__ import annotations

"""构建客户版一键运行程序包。"""

import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
DIST_ROOT = ROOT / "dist"
BUILD_ROOT = ROOT / "build"
SPEC_FILE = ROOT / "DAE程序.spec"
PACKAGE_ROOT = DIST_ROOT / "DAE程序包"
EXE_NAME = "DAE程序"
ZIP_PATH = DIST_ROOT / "DAE程序包.zip"


def _clean() -> None:
    """清理旧构建目录。"""
    for path in [BUILD_ROOT, PACKAGE_ROOT]:
        if path.exists():
            shutil.rmtree(path)
    exe_dir = DIST_ROOT / EXE_NAME
    if exe_dir.exists():
        shutil.rmtree(exe_dir)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    if SPEC_FILE.exists():
        SPEC_FILE.unlink()


def _run_pyinstaller() -> None:
    """执行 PyInstaller onedir 打包。"""
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name",
        EXE_NAME,
        "--hidden-import",
        "matplotlib.backends.backend_agg",
        "--hidden-import",
        "sklearn.metrics",
        "customer_launcher.py",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _assemble_package() -> Path:
    """组装客户最终交付目录。"""
    built_dir = DIST_ROOT / EXE_NAME
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)

    for item in built_dir.iterdir():
        target = PACKAGE_ROOT / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    data_src = ROOT / "CNN" / "data567.mat"
    if data_src.exists():
        shutil.copy2(data_src, PACKAGE_ROOT / "data567.mat")
    return PACKAGE_ROOT


def _zip_package() -> Path:
    """把客户目录再次压缩为 zip。"""
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as zf:
        for path in PACKAGE_ROOT.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(DIST_ROOT))
    return ZIP_PATH


def main() -> None:
    """客户版程序包构建入口。"""
    _clean()
    _run_pyinstaller()
    package_dir = _assemble_package()
    zip_path = _zip_package()
    print(package_dir.resolve())
    print(zip_path.resolve())


if __name__ == "__main__":
    main()
