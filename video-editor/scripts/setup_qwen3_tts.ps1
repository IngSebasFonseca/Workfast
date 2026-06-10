$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $Root "assets\tools\qwen3_tts_venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$SoxDir = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2"
$SoxExe = Join-Path $SoxDir "sox.exe"

if (!(Get-Command sox -ErrorAction SilentlyContinue) -and !(Test-Path $SoxExe)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id ChrisBagwell.SoX -e --silent --accept-package-agreements --accept-source-agreements
    } else {
        Write-Warning "No encontre SoX ni winget. Instala SoX manualmente y agregalo al PATH."
    }
}

if (Test-Path $SoxExe) {
    $env:Path = "$env:Path;$SoxDir"
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($null -eq $UserPath) { $UserPath = "" }
    if (($UserPath -split ";") -notcontains $SoxDir) {
        [Environment]::SetEnvironmentVariable("Path", ($UserPath.TrimEnd(";") + ";" + $SoxDir).TrimStart(";"), "User")
    }
}

if (!(Test-Path $Python)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Venv) | Out-Null
    py -3.11 -m venv $Venv
}

& $Python -m pip install -U pip wheel
& $Python -m pip install "setuptools<82"

# RTX 50xx/Blackwell works best with current official CUDA 12.8 PyTorch wheels.
# This isolated venv keeps WorkFast's stable DeepFilterNet/Torch stack untouched.
& $Python -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu128
& $Python -m pip install -U qwen-tts==0.1.1
& $Python -m pip install -U hf_xet
& $Python -m pip install "setuptools<82"

@'
import torch
import qwen_tts
import shutil
print("qwen_tts:", getattr(qwen_tts, "__version__", "installed"))
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("sox:", shutil.which("sox") or "not-found")
'@ | & $Python -
