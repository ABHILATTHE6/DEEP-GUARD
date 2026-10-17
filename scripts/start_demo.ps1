# Start the local DEEP-GUARD API and dashboard in separate PowerShell windows.
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$checkpointPath = Join-Path $repoRoot "artifacts\image_training_best.pt"
$frontendPath = Join-Path $repoRoot "frontend"
$viteCmd = Join-Path $frontendPath "node_modules\.bin\vite.cmd"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Python virtual environment not found at $pythonPath. Create/install the project environment first."
    exit 1
}
if (-not (Test-Path $checkpointPath)) {
    Write-Error "Trained checkpoint not found at $checkpointPath. Download image-training-best.pt from the training-checkpoints GitHub Release using docs/TRAINING.md."
    exit 1
}
if (-not (Test-Path $viteCmd)) {
    Write-Error "Frontend dependencies are missing. Run 'cd frontend; npm ci' first, then rerun this script."
    exit 1
}

$servingCheck = & $pythonPath -c "import cv2, librosa, soundfile, torch, torchvision, transformers"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Multimodal serving dependencies are missing. Run '.\\.venv\\Scripts\\python.exe -m pip install -e "".[serving,dev]""' from the repository root, then rerun this script."
    exit 1
}

function ConvertTo-PowerShellLiteral([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

$quotedRoot = ConvertTo-PowerShellLiteral $repoRoot
$quotedCheckpoint = ConvertTo-PowerShellLiteral $checkpointPath
$quotedPython = ConvertTo-PowerShellLiteral $pythonPath
$quotedFrontend = ConvertTo-PowerShellLiteral $frontendPath

$backendCommand = "Set-Location -LiteralPath $quotedRoot; `$env:DEEP_GUARD_IMAGE_CHECKPOINT = $quotedCheckpoint; & $quotedPython -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000"
$frontendCommand = "Set-Location -LiteralPath $quotedFrontend; npm run dev"

Start-Process -FilePath "powershell.exe" -WorkingDirectory $repoRoot -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand
)
Start-Sleep -Seconds 2
Start-Process -FilePath "powershell.exe" -WorkingDirectory $frontendPath -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand
)

Write-Host ""
Write-Host "DEEP-GUARD startup commands were opened in separate PowerShell windows." -ForegroundColor Green
Write-Host "Backend health: http://127.0.0.1:8000/api/v1/health"
Write-Host "Dashboard:      http://localhost:5173"
Write-Host "Keep both service windows open while demonstrating the app."
