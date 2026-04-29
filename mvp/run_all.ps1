Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$mplConfig = Join-Path $workspaceRoot '.mplconfig'
New-Item -ItemType Directory -Force $mplConfig | Out-Null
$env:MPLCONFIGDIR = $mplConfig
$pythonCandidates = @(
    (Join-Path $env:LocalAppData 'Programs\Python\Python312\python.exe'),
    (Join-Path $env:LocalAppData 'Programs\Python\Python313\python.exe'),
    'python'
)
$pythonExe = $pythonCandidates | Where-Object { $_ -eq 'python' -or (Test-Path $_) } | Select-Object -First 1

& $pythonExe src/generate_synthetic_data.py
& $pythonExe src/train_models.py
& $pythonExe src/make_forecast.py
& $pythonExe src/build_inventory_plan.py

Write-Host "MVP artifacts created under mvp/artifacts"
Write-Host "Launch demo with: $pythonExe app.py"
