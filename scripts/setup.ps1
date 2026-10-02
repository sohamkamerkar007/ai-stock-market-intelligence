param([switch]$SkipData, [switch]$SkipModels)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '..\setup.ps1') -SkipData:$SkipData -SkipModels:$SkipModels
