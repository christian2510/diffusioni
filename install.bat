@echo off
chcp 65001 > nul
setlocal

echo ===========================================
echo Diffusioni v.0.1 Alpha - Installationsskript
echo ===========================================
echo.

rem --- Schritt 1: Python und virtuelle Umgebung pruefen/einrichten ---
echo Pruefe auf Python-Installation und richte virtuelle Umgebung ein...
echo.

python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python wurde nicht gefunden.
    echo "Bitte installieren Sie Python (empfohlen: Miniconda) von:"
    echo https://docs.conda.io/en/latest/miniconda.html
    echo Starten Sie dieses Skript danach erneut.
    pause
    exit /b 1
)

if not exist "diffusioni_env" (
    echo Erstelle virtuelle Umgebung 'diffusioni_env'...
    python -m venv diffusioni_env
    if %errorlevel% neq 0 (
        echo FEHLER: Konnte virtuelle Umgebung nicht erstellen. Bitte pruefen Sie Ihre Python-Installation und Berechtigungen.
        pause
        exit /b 1
    )
) else (
    echo Virtuelle Umgebung 'diffusioni_env' existiert bereits.
)

echo Aktiviere virtuelle Umgebung...
call diffusioni_env\Scripts\activate
if %errorlevel% neq 0 (
    echo FEHLER: Konnte virtuelle Umgebung nicht aktivieren.
    pause
    exit /b 1
)
echo Virtuelle Umgebung aktiviert.

rem --- Schritt 2: Allgemeine Python-Abhaengigkeiten installieren ---
echo.
echo Installiere Kern-Abhaengigkeiten (customtkinter, Pillow, numpy, safetensors, pyperclip)...
pip install customtkinter Pillow numpy safetensors pyperclip
if %errorlevel% neq 0 (
    echo FEHLER: Konnte Kern-Abhaengigkeiten nicht installieren. Bitte pruefen Sie Ihre Internetverbindung.
    pause
    exit /b 1
)
echo Kern-Abhaengigkeiten installiert.

rem --- Schritt 3: PyTorch und Diffusers installieren (WICHTIG!) ---
echo.
echo ==================================================================================
echo WICHTIG: PyTorch-Installation (manuell anpassen!)
echo ==================================================================================
echo Die korrekte PyTorch-Installation haengt von Ihrer NVIDIA CUDA-Version ab.
echo "Bitte besuchen Sie die offizielle PyTorch-Website und waehlen Sie Ihre Konfiguration:"
echo https://pytorch.org/get-started/locally/
echo.
echo Kopieren Sie den Teil des Installationsbefehls, der NACH "pip install " kommt.
echo Fuegen Sie diesen Teil in der naechsten Zeile ein und druecken Sie ENTER.
echo.
echo Beispiel fuer CUDA 12.1: torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo Beispiel fuer CPU-only: torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo ==================================================================================
echo.
set /p PYTORCH_INSTALL_ARGS="Bitte fuegen Sie den PyTorch-Installationsbefehl hier ein: "
if "%PYTORCH_INSTALL_ARGS%"=="" (
    echo FEHLER: Kein PyTorch-Installationsbefehl eingegeben.
    pause
    exit /b 1
)
pip install %PYTORCH_INSTALL_ARGS%
if %errorlevel% neq 0 (
    echo FEHLER: Konnte PyTorch nicht installieren. Der eingegebene Befehl war moeglicherweise falsch oder es gab ein Problem mit Ihrer CUDA-Installation.
    echo Bitte ueberpruefen Sie die PyTorch-Website und Ihre Treiber.
    pause
    exit /b 1
)
echo PyTorch installiert.

rem --- Schritt 4: Diffusers und xformers installieren ---
echo.
echo Installiere diffusers und accelerate...
pip install --upgrade diffusers transformers accelerate
if %errorlevel% neq 0 (
    echo FEHLER: Konnte diffusers/accelerate nicht installieren.
    pause
    exit /b 1
)
echo diffusers und accelerate installiert.

echo.
echo Versuche, xformers zu installieren (optional, nur fuer NVIDIA GPUs)...
pip install xformers
if %errorlevel% neq 0 (
    echo WARNUNG: xformers konnte nicht installiert werden. Dies ist optional, kann aber die Performance auf NVIDIA GPUs verbessern.
    echo Das Programm wird ohne xformers fortgesetzt.
) else (
    echo xformers erfolgreich installiert.
)

rem --- Schritt 5: Modelle- und Output-Ordner sicherstellen ---
echo.
echo Stelle sicher, dass 'models' und 'output' Ordner existieren...
if not exist "models" mkdir models
if not exist "output" mkdir output
echo Ordner sind bereit.

echo.
echo ===========================================
echo Installation abgeschlossen!
echo ===========================================
echo.
echo Um Diffusioni zu starten, fuehren Sie aus:
echo python diffusioni.py
echo.
echo Um den CPU-Modus zu erzwingen (langsamer):
echo python diffusioni.py /cpu
echo.
pause
endlocal
