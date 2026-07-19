import os
import sys
import platform
import subprocess
import shutil

EIGEN_VERSION = "2.8.0"
EIGEN_CODENAME = "Mars"


def run_cmd(cmd, shell=False):
    print(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, shell=shell)
    if res.returncode != 0:
        print(f"Error: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def build_pyinstaller(system):
    """Build standalone executable using PyInstaller."""
    sep = ';' if system == 'windows' else ':'
    pyinstaller_args = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "eigen",
        f"--add-data=stdlib{sep}stdlib",
        "--exclude-module=torch",
        "--exclude-module=torchvision",
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=sympy",
        "--exclude-module=pandas",
        "--exclude-module=cv2",
        "--exclude-module=pygame",
        "--exclude-module=onnxruntime",
        "--exclude-module=transformers",
        "--exclude-module=yt_dlp",
        "--exclude-module=altair",
        "--exclude-module=sqlalchemy",
        "--exclude-module=pdfminer",
        "--exclude-module=pypdfium2",
        "--exclude-module=pypdfium2_raw",
        "--exclude-module=soundfile",
        "--exclude-module=librosa",
        "--exclude-module=av",
        "--exclude-module=bitsandbytes",
        "src/main.py"
    ]
    run_cmd(pyinstaller_args)


def build_windows_installer(dist_dir):
    """Build Windows installer using Inno Setup with fallback to standalone exe."""
    win_exe = os.path.join(dist_dir, "eigen.exe")

    if not os.path.exists(win_exe):
        print("Error: PyInstaller build failed to generate output.", file=sys.stderr)
        sys.exit(1)

    installer_script = os.path.join("installer", "eigen_setup.iss")
    iscc_path = shutil.which("iscc")

    if iscc_path and os.path.exists(installer_script):
        icon_path = os.path.join("installer", "eigen_icon.ico")
        if not os.path.exists(icon_path):
            print("Warning: eigen_icon.ico not found. Removing SetupIconFile from script...")

        print(f"Building Inno Setup installer from {installer_script}...")
        run_cmd([iscc_path, installer_script], shell=True)

        setup_exe = os.path.join(dist_dir, f"Eigen-{EIGEN_VERSION}-Setup-Windows-x64.exe")
        if os.path.exists(setup_exe):
            print(f"Windows Inno Setup installer successfully created at: {setup_exe}")
            return
        else:
            print("Warning: Inno Setup compilation did not produce output. Falling back to standalone exe.")

    target_win = os.path.join(dist_dir, f"Eigen-{EIGEN_VERSION}-Windows-x64.exe")
    if os.path.exists(target_win):
        os.remove(target_win)
    os.rename(win_exe, target_win)
    print(f"Windows standalone installer created at: {target_win}")
    print("Note: Install Inno Setup (https://jrsoftware.org/isinfo.php) for a full GUI wizard installer.")


def build_linux_appimage(dist_dir):
    """Build Linux AppImage with tar.gz fallback."""
    os.environ["ARCH"] = "x86_64"
    os.environ["APPIMAGE_EXTRACT_AND_RUN"] = "1"

    app_dir = os.path.abspath("Eigen.AppDir")
    usr_bin = os.path.join(app_dir, "usr", "bin")
    os.makedirs(usr_bin, exist_ok=True)

    shutil.copy(os.path.join(dist_dir, "eigen"), os.path.join(usr_bin, "eigen"))

    apprun_path = os.path.join(app_dir, "AppRun")
    with open(apprun_path, "w") as f:
        f.write(
            "#!/bin/sh\n"
            "SELF=$(readlink -f \"$0\")\n"
            "HERE=$(dirname \"$SELF\")\n"
            "exec \"$HERE/usr/bin/eigen\" \"$@\"\n"
        )
    os.chmod(apprun_path, 0o755)

    desktop_path = os.path.join(app_dir, "eigen.desktop")
    with open(desktop_path, "w") as f:
        f.write("[Desktop Entry]\nType=Application\nName=Eigen\nExec=eigen\nIcon=eigen\nCategories=Development;\n")

    import struct, zlib
    def _make_minimal_png():
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0)
        ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
        raw_row = b'\x00\x00\x00\x00\x00'
        idat_data = zlib.compress(raw_row)
        idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + idat_data) & 0xffffffff)
        iend_crc = struct.pack('>I', zlib.crc32(b'IEND') & 0xffffffff)
        return (b'\x89PNG\r\n\x1a\n'
                + struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + ihdr_crc
                + struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + idat_crc
                + b'\x00\x00\x00\x00IEND' + iend_crc)
    with open(os.path.join(app_dir, "eigen.png"), "wb") as f:
        f.write(_make_minimal_png())

    target_appimage = os.path.join(dist_dir, f"Eigen-{EIGEN_VERSION}-Linux.AppImage")
    appimagetool_path = shutil.which("appimagetool")

    if not appimagetool_path:
        print("appimagetool not found. Downloading...")
        try:
            url = "https://github.com/AppImage/AppImageKit/releases/download/13/appimagetool-x86_64.AppImage"
            wget_target = os.path.abspath("appimagetool-x86_64.AppImage")
            result = subprocess.run(["curl", "-L", "-f", "-o", wget_target, url])
            if result.returncode != 0:
                print(f"curl download failed with code {result.returncode}")
                appimagetool_path = None
            else:
                file_size = os.path.getsize(wget_target)
                if file_size < 1000:
                    print(f"appimagetool download seems too small ({file_size} bytes), likely failed")
                    appimagetool_path = None
                else:
                    os.chmod(wget_target, 0o755)
                    appimagetool_path = wget_target
        except Exception as e:
            print(f"Could not download appimagetool: {e}")
            appimagetool_path = None

    appimage_ok = False
    if appimagetool_path:
        try:
            extract_dir = os.path.abspath("appimagetool-extracted")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            print("Extracting appimagetool...")
            subprocess.run([appimagetool_path, "--appimage-extract"],
                           cwd=extract_dir, check=True)
            extracted_apprun = os.path.join(extract_dir, "squashfs-root", "AppRun")
            print(f"Building AppImage with: {extracted_apprun}")
            run_cmd([extracted_apprun, app_dir, target_appimage])
            appimage_ok = True
            print(f"Linux AppImage successfully created at: {target_appimage}")
        except (SystemExit, subprocess.CalledProcessError, OSError, Exception) as e:
            print(f"appimagetool failed: {e}. Falling back to tar.gz packaging...")

    if not appimage_ok:
        import tarfile
        tar_path = os.path.join(dist_dir, f"Eigen-{EIGEN_VERSION}-Linux.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(os.path.join(dist_dir, "eigen"), arcname="eigen")
        shutil.copy(tar_path, target_appimage)
        print(f"Linux tar.gz fallback created at: {target_appimage}")


def build_macos_pkg(dist_dir):
    """Build macOS .pkg installer."""
    pkg_root = os.path.abspath("pkgroot")
    usr_local_bin = os.path.join(pkg_root, "usr", "local", "bin")
    os.makedirs(usr_local_bin, exist_ok=True)

    shutil.copy(os.path.join(dist_dir, "eigen"), os.path.join(usr_local_bin, "eigen"))

    target_pkg = os.path.join(dist_dir, f"Eigen-{EIGEN_VERSION}-macOS.pkg")
    pkgbuild_args = [
        "pkgbuild",
        "--identifier", "com.eigenresearch.eigen",
        "--version", EIGEN_VERSION,
        "--root", pkg_root,
        target_pkg
    ]

    if shutil.which("pkgbuild"):
        run_cmd(pkgbuild_args)
        print(f"macOS package successfully created at: {target_pkg}")
    else:
        print(
            "pkgbuild command not found. macOS installer package building skipped. "
            "Standalone binary is at dist/eigen."
        )


def main():
    system = platform.system().lower()
    print(f"Starting Eigen {EIGEN_VERSION} \"{EIGEN_CODENAME}\" build process on platform: {system}")

    try:
        import PyInstaller  # noqa: F401  (availability check)
        print("PyInstaller is already installed.")
    except ImportError:
        print("Installing PyInstaller...")
        run_cmd([sys.executable, "-m", "pip", "install", "pyinstaller"])

    for folder in ['build', 'dist', 'Eigen.AppDir', 'pkgroot']:
        if os.path.exists(folder):
            print(f"Removing old {folder} directory...")
            shutil.rmtree(folder)

    build_pyinstaller(system)

    dist_dir = os.path.abspath("dist")
    os.makedirs(dist_dir, exist_ok=True)

    if system == 'windows':
        build_windows_installer(dist_dir)
    elif system == 'linux':
        build_linux_appimage(dist_dir)
    elif system == 'darwin':
        build_macos_pkg(dist_dir)
    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)


if __name__ == '__main__':
    main()
