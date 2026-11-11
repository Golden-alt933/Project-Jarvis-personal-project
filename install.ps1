# Jarvis installer for Windows (PowerShell 5.1 or newer).
#
#   irm https://raw.githubusercontent.com/Golden-alt933/Project-Jarvis-personal-project/main/install.ps1 | iex
#
# Options cannot be passed through `irm | iex`; set environment variables first instead:
#   $env:JARVIS_REF = 'v0.1.0'; $env:JARVIS_MINIMAL = '1'; $env:JARVIS_GPU = '1'; $env:JARVIS_NO_SETUP = '1'
# When the file is saved and run directly the flags work too:
#   .\install.ps1 --minimal --gpu --ref main
#
# What it does: makes sure `uv` is installed, installs Jarvis as a uv tool from this
# repository at a pinned release, and runs `jarvis --setup`. Re-running it upgrades.

$ErrorActionPreference = 'Stop'

# ---- release to install: bump this one line when cutting a release ------------------
$DefaultRef = 'v0.1.0'
# --------------------------------------------------------------------------------------

$RepoUrl = 'https://github.com/Golden-alt933/Project-Jarvis-personal-project'
$UvInstallerUrl = 'https://astral.sh/uv/install.ps1'

$Ref = if ($env:JARVIS_REF) { $env:JARVIS_REF } else { $DefaultRef }
$Minimal = ($env:JARVIS_MINIMAL -eq '1')
$Gpu = ($env:JARVIS_GPU -eq '1')

$i = 0
while ($i -lt $args.Count) {
    switch -Regex ($args[$i]) {
        '^(--|-)minimal$' { $Minimal = $true }
        '^(--|-)gpu$' { $Gpu = $true }
        '^(--|-)ref$' {
            $i++
            if ($i -ge $args.Count) { throw '--ref needs a value' }
            $Ref = $args[$i]
        }
        '^(--|-)(h|help)$' {
            Write-Host 'Usage: install.ps1 [--minimal] [--gpu] [--ref <tag|branch|commit>]'
            return
        }
        default { throw "unknown option: $($args[$i])" }
    }
    $i++
}

function Say($text) { Write-Host "==> $text" }

# ---- 1. uv ----------------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say 'uv (the Python package manager Jarvis installs with) is not installed.'
    Say "Installing it with Astral's official installer: $UvInstallerUrl"
    Invoke-RestMethod $UvInstallerUrl | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv was installed but is not on PATH; open a new terminal and run this script again'
    }
}
Say "Using $(uv --version)"

# ---- 2. Jarvis ------------------------------------------------------------------------
$extras = @()
if (-not $Minimal) { $extras += 'all' }
if ($Gpu) { $extras += 'gpu' }
$name = if ($extras.Count -gt 0) { 'jarvis[' + ($extras -join ',') + ']' } else { 'jarvis' }
$spec = "$name @ git+${RepoUrl}@${Ref}"

Say "Installing Jarvis $Ref (re-running this upgrades it)"
& uv tool install --python 3.13 --force --reinstall-package jarvis --refresh-package jarvis $spec
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed (exit code $LASTEXITCODE)" }

$binDir = (& uv tool dir --bin 2>$null)
if (-not $binDir) { $binDir = "$env:USERPROFILE\.local\bin" }
$jarvis = Join-Path $binDir 'jarvis.exe'
if (-not (Test-Path $jarvis)) {
    $found = Get-Command jarvis -ErrorAction SilentlyContinue
    if ($found) { $jarvis = $found.Source } else { throw "the install finished but jarvis.exe was not found in $binDir" }
}
if (-not (Get-Command jarvis -ErrorAction SilentlyContinue)) {
    Write-Warning "$binDir is not on your PATH yet. Run 'uv tool update-shell' and open a new terminal."
}
& $jarvis --version
if ($LASTEXITCODE -ne 0) { throw 'jarvis --version failed' }

# ---- 3. first run ----------------------------------------------------------------------
$interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected -and -not $env:JARVIS_NO_SETUP -and -not $env:CI
if ($interactive) {
    Say "Starting first-run setup (Ctrl+C to skip; run 'jarvis --setup' any time)"
    & $jarvis --setup
}
else {
    Write-Host @'

Next steps:
  jarvis --setup     choose a provider and store your API key (free key: https://build.nvidia.com)
  jarvis --doctor    check that everything works
  jarvis             start chatting

Uninstall: uv tool uninstall jarvis   (settings and memory stay; see the README for where they live)
'@
}
