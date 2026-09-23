$ErrorActionPreference = "Stop"
$python = "E:\conda\envs\yolo\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
Set-Location $PSScriptRoot
& $python prelabel.py @args
