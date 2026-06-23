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
        "src/main.py"
    ]
    run_cmd(pyinstaller_args)

    dist_dir = os.path.abspath("dist")
    os.makedirs(dist_dir, exist_ok=True)

    # 3. Platform specific installer packaging
    if system == 'windows':
        # Standalone executable is the installer
        win_exe = os.path.join(dist_dir, "eigen.exe")
        target_win = os.path.join(dist_dir, "Eigen-2.3-Windows-x64.exe")
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
            
        # Create a valid minimal 1x1 PNG icon to satisfy appimagetool validation
        with open(os.path.join(app_dir, "eigen.png"), "wb") as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB')
            
        # Try to use appimagetool to build AppImage
        appimagetool_path = shutil.which("appimagetool")
        if not appimagetool_path:
            # Try to download appimagetool
            print("appimagetool not found. Downloading appimagetool...")
            try:
                url = "https://github.com/AppImage/AppImageKit/releases/download/13/appimagetool-x86_64.AppImage"
                wget_target = os.path.abspath("appimagetool-x86_64.AppImage")
                run_cmd(["curl", "-L", "-o", wget_target, url])
                os.chmod(wget_target, 0o755)
                appimagetool_path = wget_target
            except Exception as e:
                print(f"Could not download appimagetool: {e}. Standalone binary is left at dist/eigen.")
                return
                
        # Build AppImage
        target_appimage = os.path.join(dist_dir, "Eigen-2.3-Linux.AppImage")
        if appimagetool_path.endswith(".AppImage"):
            print("Extracting appimagetool to avoid FUSE issues...")
            run_cmd([appimagetool_path, "--appimage-extract"])
            appimagetool_path = os.path.abspath("squashfs-root/AppRun")
            
        run_cmd([appimagetool_path, app_dir, target_appimage])
        print(f"Linux AppImage successfully created at: {target_appimage}")

    elif system == 'darwin':
        # Create macOS .pkg installer using pkgbuild
        pkg_root = os.path.abspath("pkgroot")
        usr_local_bin = os.path.join(pkg_root, "usr", "local", "bin")
        os.makedirs(usr_local_bin, exist_ok=True)
        
        shutil.copy(os.path.join(dist_dir, "eigen"), os.path.join(usr_local_bin, "eigen"))
        
        target_pkg = os.path.join(dist_dir, "Eigen-2.3-macOS.pkg")
        pkgbuild_args = [
            "pkgbuild",
            "--identifier", "com.eigenresearch.eigen",
            "--version", "2.3.0",
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
