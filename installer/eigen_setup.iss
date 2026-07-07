; Eigen Programming Language - Inno Setup Script v2.7.0
; Build with: iscc installer\eigen_setup.iss
; Requires: Inno Setup 6.x (https://jrsoftware.org/isinfo.php)

#define EigenVersion "2.7.0"
#define EigenCodename "Meridian"

[Setup]
AppName=Eigen Programming Language
AppVersion={#EigenVersion}
AppVerName=Eigen {#EigenVersion} "{#EigenCodename}"
AppPublisher=Eigen Research
AppPublisherURL=https://github.com/Eigenresearch/Eigen
AppSupportURL=https://github.com/Eigenresearch/Eigen/issues
AppUpdatesURL=https://github.com/Eigenresearch/Eigen/releases
DefaultDirName={autopf}\Eigen
DefaultGroupName=Eigen
OutputBaseFilename=Eigen-{#EigenVersion}-Setup-Windows-x64
; SetupIconFile=eigen_icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=admin
ChangesEnvironment=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\dist
UninstallDisplayIcon={app}\eigen.exe
LicenseFile=LICENSE.txt
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Types]
Name: "full"; Description: "Full Installation"
Name: "compact"; Description: "Compact Installation"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Eigen Core (Compiler, VM, Runtime)"; Types: full compact custom; Flags: fixed
Name: "stdlib"; Description: "Standard Library (math, collections, io...)"; Types: full compact
Name: "examples"; Description: "Quantum Examples (Bell, GHZ, Grover, QFT, Shor)"; Types: full
Name: "gpu"; Description: "GPU Acceleration (CUDA/Vulkan kernels)"; Types: full
Name: "native"; Description: "Native Rust Extensions (native parser, optimizer)"; Types: full
Name: "vscode"; Description: "VS Code Extension"; Types: full

[Tasks]
Name: "desktopicon"; Description: "Create Desktop shortcut"; GroupDescription: "Additional shortcuts:"
Name: "addtopath"; Description: "Add to PATH (available after restart)"; GroupDescription: "Other:"; Flags: checkedonce
Name: "fileassoc"; Description: "Register Eigen as editor for .eig files"; GroupDescription: "Other:"; Flags: checkedonce
Name: "contextmenu"; Description: "Add 'Open with Eigen' to file context menu"; GroupDescription: "Other:"
Name: "dircontextmenu"; Description: "Add 'Open with Eigen' to directory context menu"; GroupDescription: "Other:"

[Files]
Source: "..\dist\eigen.exe"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "..\stdlib\*"; DestDir: "{app}\stdlib"; Flags: ignoreversion recursesubdirs; Components: stdlib
Source: "..\examples\*"; DestDir: "{app}\examples"; Flags: ignoreversion recursesubdirs; Components: examples
Source: "..\native\rust\target\release\eigen_native.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: native
Source: "..\vscode-extension\*"; DestDir: "{app}\vscode-extension"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist; Components: vscode

[Icons]
Name: "{group}\Eigen"; Filename: "{app}\eigen.exe"
Name: "{group}\Eigen Documentation"; Filename: "https://github.com/Eigenresearch/Eigen"
Name: "{group}\Uninstall Eigen"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Eigen"; Filename: "{app}\eigen.exe"; Tasks: desktopicon

[Registry]
Root: HKCR; Subkey: ".eig"; ValueType: string; ValueName: ""; ValueData: "EigenSourceFile"; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKCR; Subkey: "EigenSourceFile"; ValueType: string; ValueName: ""; ValueData: "Eigen Source File"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKCR; Subkey: "EigenSourceFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\eigen.exe,0"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKCR; Subkey: "EigenSourceFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\eigen.exe"" run ""%1"""; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKCU; Subkey: "Software\Classes\*\shell\OpenWithEigen"; ValueType: string; ValueName: ""; ValueData: "Open with Eigen"; Flags: uninsdeletekey; Tasks: contextmenu
Root: HKCU; Subkey: "Software\Classes\*\shell\OpenWithEigen\command"; ValueType: string; ValueName: ""; ValueData: """{app}\eigen.exe"" run ""%1"""; Flags: uninsdeletekey; Tasks: contextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\OpenWithEigen"; ValueType: string; ValueName: ""; ValueData: "Open with Eigen"; Flags: uninsdeletekey; Tasks: dircontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\OpenWithEigen\command"; ValueType: string; ValueName: ""; ValueData: """{app}\eigen.exe"" run ""%1"""; Flags: uninsdeletekey; Tasks: dircontextmenu

[Run]
Filename: "{app}\eigen.exe"; Parameters: "doctor"; Description: "Run eigen doctor (verify installation)"; Flags: postinstall nowait skipifsilent unchecked
Filename: "https://github.com/Eigenresearch/Eigen"; Description: "Open documentation"; Flags: postinstall nowait skipifsilent unchecked

[UninstallRun]
Filename: "{app}\eigen.exe"; Parameters: "doctor"; Flags: nowait

[Code]
function NeedsAddPath(InstallDir: string): boolean;
var
  Path: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path) then
    begin
      Result := True;
      exit;
    end;
  if Pos(';' + InstallDir + ';', ';' + Path + ';') > 0 then
    Result := False
  else
    Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Path: string;
begin
  if CurStep = ssPostInstall then
  begin
    if IsTaskSelected('addtopath') and NeedsAddPath(ExpandConstant('{app}')) then
    begin
      if RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path) then
      begin
        Path := Path + ';' + ExpandConstant('{app}');
        RegWriteStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path);
      end
      else
      begin
        RegWriteStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', ExpandConstant('{app}'));
      end;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Path: string;
  AppDir: string;
  PathtoRemove: string;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppDir := ExpandConstant('{app}');
    if RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path) then
    begin
      PathtoRemove := ';' + AppDir;
      if Pos(PathtoRemove, Path) > 0 then
      begin
        StringChange(Path, PathtoRemove, '');
        RegWriteStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path);
      end
      else if Pos(AppDir + ';', Path) = 1 then
      begin
        StringChange(Path, AppDir + ';', '');
        RegWriteStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path);
      end
      else if Pos(AppDir, Path) = 1 then
      begin
        StringChange(Path, AppDir, '');
        RegWriteStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path);
      end;
    end;
  end;
end;
