import os
import sys
import platform
import subprocess
import shutil

def run_cmd(cmd, shell=False):
    print(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, shell=shell)
    if res.returncode != 0:
        print(f"Error: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)

def main():
    system = platform.system().lower()
    print(f"Starting build process on platform: {system}")
    
    # 1. Ensure pyinstaller is installed
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
    except ImportError:
        print("Installing PyInstaller...")
        run_cmd([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Clean previous build artifacts
    for folder in ['build', 'dist', 'Eigen.AppDir', 'pkgroot']:
        if os.path.exists(folder):
            print(f"Removing old {folder} directory...")
            shutil.rmtree(folder)

    # 2. Build the core standalone executable using PyInstaller
    # Include stdlib folder as data
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

    dist_dir = os.path.abspath("dist")
    os.makedirs(dist_dir, exist_ok=True)

    # 3. Platform specific installer packaging
    if system == 'windows':
        # Standalone executable is the installer
        win_exe = os.path.join(dist_dir, "eigen.exe")
        target_win = os.path.join(dist_dir, "Eigen-2.5-Windows-x64.exe")
        if os.path.exists(win_exe):
            if os.path.exists(target_win):
                os.remove(target_win)
            os.rename(win_exe, target_win)
            print(f"Windows installer successfully created at: {target_win}")
        else:
            print("Error: PyInstaller build failed to generate output.", file=sys.stderr)
            sys.exit(1)

    elif system == 'linux':
        # Create AppImage
        os.environ["ARCH"] = "x86_64"
        # APPIMAGE_EXTRACT_AND_RUN lets AppImages run without FUSE (required on Ubuntu 24.04+)
        os.environ["APPIMAGE_EXTRACT_AND_RUN"] = "1"

        app_dir = os.path.abspath("Eigen.AppDir")
        usr_bin = os.path.join(app_dir, "usr", "bin")
        os.makedirs(usr_bin, exist_ok=True)
        
        # Copy binary
        shutil.copy(os.path.join(dist_dir, "eigen"), os.path.join(usr_bin, "eigen"))
        
        # Create AppRun
        apprun_path = os.path.join(app_dir, "AppRun")
        with open(apprun_path, "w") as f:
            f.write("#!/bin/sh\nSELF=$(readlink -f \"$0\")\nHERE=$(dirname \"$SELF\")\nexec \"$HERE/usr/bin/eigen\" \"$@\"\n")
        os.chmod(apprun_path, 0o755)
        
        # Create Desktop Entry
        desktop_path = os.path.join(app_dir, "eigen.desktop")
        with open(desktop_path, "w") as f:
            f.write("[Desktop Entry]\nType=Application\nName=Eigen\nExec=eigen\nIcon=eigen\nCategories=Development;\n")
            
        # Create a valid minimal 1x1 PNG icon
        import struct, zlib
        def _make_minimal_png():
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0)
            ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
            raw_row = b'\x00\x00\x00\x00\x00'  # filter=None, RGBA=0,0,0,0
            idat_data = zlib.compress(raw_row)
            idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + idat_data) & 0xffffffff)
            iend_crc = struct.pack('>I', zlib.crc32(b'IEND') & 0xffffffff)
            return (b'\x89PNG\r\n\x1a\n'
                    + struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + ihdr_crc
                    + struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + idat_crc
                    + b'\x00\x00\x00\x00IEND' + iend_crc)
        with open(os.path.join(app_dir, "eigen.png"), "wb") as f:
            f.write(_make_minimal_png())
            
        # Download appimagetool
        target_appimage = os.path.join(dist_dir, "Eigen-2.5-Linux.AppImage")
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
                # Extract appimagetool to avoid FUSE dependency on CI
                extract_dir = os.path.abspath("appimagetool-extracted")
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                os.makedirs(extract_dir, exist_ok=True)
                print(f"Extracting appimagetool...")
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
            # Fallback: create a tar.gz with the standalone binary
            import tarfile
            target_appimage = os.path.join(dist_dir, "Eigen-2.5-Linux.AppImage")
            tar_path = os.path.join(dist_dir, "Eigen-2.5-Linux.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(os.path.join(dist_dir, "eigen"), arcname="eigen")
            # Copy tar.gz as the AppImage artifact path so the workflow picks it up
            shutil.copy(tar_path, target_appimage)
            print(f"Linux tar.gz fallback created at: {target_appimage}")

    elif system == 'darwin':
        # Create macOS .pkg installer using pkgbuild
        pkg_root = os.path.abspath("pkgroot")
        usr_local_bin = os.path.join(pkg_root, "usr", "local", "bin")
        os.makedirs(usr_local_bin, exist_ok=True)
        
        shutil.copy(os.path.join(dist_dir, "eigen"), os.path.join(usr_local_bin, "eigen"))
        
        target_pkg = os.path.join(dist_dir, "Eigen-2.5-macOS.pkg")
        pkgbuild_args = [
            "pkgbuild",
            "--identifier", "com.eigenresearch.eigen",
            "--version", "2.5.0",
            "--root", pkg_root,
            target_pkg
        ]
        
        # Check if pkgbuild is available
        if shutil.which("pkgbuild"):
            run_cmd(pkgbuild_args)
            print(f"macOS package successfully created at: {target_pkg}")
        else:
            print("pkgbuild command not found. macOS installer package building skipped. Standalone binary is at dist/eigen.")

if __name__ == '__main__':
    main()

