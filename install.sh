#!/usr/bin/env bash
# Jarvis installer for Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/Golden-alt933/Project-Jarvis-personal-project/main/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --minimal        # pass options after "--"
#
# What it does: makes sure `uv` is installed, installs Jarvis as a uv tool from this
# repository at a pinned release, checks for system libraries and helper programs, and
# runs `jarvis --setup`. Re-running it upgrades. It never runs sudo unless you pass
# --system-deps, and then it prints the command first.
set -euo pipefail

# ---- release to install: bump this one line when cutting a release ------------------
JARVIS_DEFAULT_REF="v0.2.1"
# --------------------------------------------------------------------------------------

REPO_URL="https://github.com/Golden-alt933/Project-Jarvis-personal-project"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"

JARVIS_REF="${JARVIS_REF:-$JARVIS_DEFAULT_REF}"
MINIMAL=0
GPU=0
SYSTEM_DEPS=0

say() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: install.sh [options]

  --minimal        install without the optional extras (text mode only)
  --gpu            also install the NVIDIA CUDA libraries for GPU speech recognition
  --ref <tag|branch|commit>
                   version to install (default: ${JARVIS_DEFAULT_REF}; env JARVIS_REF)
  --system-deps    run the system package install command (uses sudo; asks you first)
  -h, --help       show this help

Environment: JARVIS_REF, JARVIS_NO_SETUP=1 (do not start 'jarvis --setup' at the end).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --minimal) MINIMAL=1 ;;
    --gpu) GPU=1 ;;
    --system-deps) SYSTEM_DEPS=1 ;;
    --ref)
      [ $# -ge 2 ] || die "--ref needs a value"
      JARVIS_REF="$2"
      shift
      ;;
    --ref=*) JARVIS_REF="${1#--ref=}" ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

[ "$(uname -s)" = "Linux" ] || die "this installer is for Linux; on Windows use install.ps1"
[ -n "$JARVIS_REF" ] || die "the version to install is empty"

# ---- 1. uv ----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "uv (the Python package manager Jarvis installs with) is not installed."
  say "Installing it with Astral's official installer: ${UV_INSTALLER_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$UV_INSTALLER_URL" | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$UV_INSTALLER_URL" | sh
  else
    die "need curl or wget to download uv"
  fi
  export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv was installed but is not on PATH; open a new terminal and run this script again"
fi
say "Using $(uv --version)"

# ---- 2. system libraries and helpers ---------------------------------------------------
LDCONFIG="$(command -v ldconfig || echo /sbin/ldconfig)"
have_lib() { "$LDCONFIG" -p 2>/dev/null | grep "$1" >/dev/null; }

detect_manager() {
  local ids=""
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    ids="$(. /etc/os-release && printf '%s %s' "${ID:-}" "${ID_LIKE:-}")"
  fi
  for id in $ids; do
    case "$id" in
      debian | ubuntu | linuxmint | pop | raspbian | kali | elementary) echo apt; return ;;
      fedora | rhel | centos | rocky | almalinux | nobara) echo dnf; return ;;
      arch | manjaro | endeavouros | cachyos) echo pacman; return ;;
      opensuse* | suse | sles) echo zypper; return ;;
    esac
  done
  echo ""
}

MISSING=()
# need <check-kind> <name> <apt> <dnf> <pacman> <zypper>
need() {
  local kind="$1" name="$2" apt="$3" dnf="$4" pacman="$5" zypper="$6" found=1
  case "$kind" in
    cmd) command -v "$name" >/dev/null 2>&1 || found=0 ;;
    lib) have_lib "$name" || found=0 ;;
  esac
  [ "$found" = 1 ] && return 0
  case "$MANAGER" in
    apt) MISSING+=("$apt") ;;
    dnf) MISSING+=("$dnf") ;;
    pacman) MISSING+=("$pacman") ;;
    zypper) MISSING+=("$zypper") ;;
    *) MISSING+=("$name") ;;
  esac
}

MANAGER="$(detect_manager)"
need cmd xdg-open xdg-utils xdg-utils xdg-utils xdg-utils
need cmd wmctrl wmctrl wmctrl wmctrl wmctrl
need cmd playerctl playerctl playerctl playerctl playerctl
if ! command -v xclip >/dev/null 2>&1 && ! command -v xsel >/dev/null 2>&1 && ! command -v wl-copy >/dev/null 2>&1; then
  if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
    need cmd wl-copy wl-clipboard wl-clipboard wl-clipboard wl-clipboard
  else
    need cmd xclip xclip xclip xclip xclip
  fi
fi
if [ "$MINIMAL" = 0 ]; then
  need lib libportaudio.so.2 libportaudio2 portaudio portaudio libportaudio2
  need lib libxcb-cursor.so.0 libxcb-cursor0 xcb-util-cursor xcb-util-cursor libxcb-cursor0
  need lib libxkbcommon-x11.so.0 libxkbcommon-x11-0 libxkbcommon-x11 libxkbcommon-x11 libxkbcommon-x11-0
  need lib libEGL.so.1 libegl1 mesa-libEGL libglvnd Mesa-libEGL1
  # the rest of what Qt's xcb platform plugin (the HUD) loads
  need lib libxcb-icccm.so.4 libxcb-icccm4 xcb-util-wm xcb-util-wm libxcb-icccm4
  need lib libxcb-image.so.0 libxcb-image0 xcb-util-image xcb-util-image libxcb-image0
  need lib libxcb-keysyms.so.1 libxcb-keysyms1 xcb-util-keysyms xcb-util-keysyms libxcb-keysyms1
  need lib libxcb-render-util.so.0 libxcb-render-util0 xcb-util-renderutil xcb-util-renderutil libxcb-render-util0
  need lib libxcb-xinerama.so.0 libxcb-xinerama0 libxcb libxcb libxcb-xinerama0
  need lib libxcb-shape.so.0 libxcb-shape0 libxcb libxcb libxcb-shape0
  need lib libxcb-randr.so.0 libxcb-randr0 libxcb libxcb libxcb-randr0
  need lib libxcb-xfixes.so.0 libxcb-xfixes0 libxcb libxcb libxcb-xfixes0
  need lib libxcb-sync.so.1 libxcb-sync1 libxcb libxcb libxcb-sync1
  need lib libxcb-xkb.so.1 libxcb-xkb1 libxcb libxcb libxcb-xkb1
  need lib libxcb-xinput.so.0 libxcb-xinput0 libxcb libxcb libxcb-xinput0
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
  case "$MANAGER" in
    apt) INSTALL_CMD=(apt-get install -y "${MISSING[@]}") ;;
    dnf) INSTALL_CMD=(dnf install -y "${MISSING[@]}") ;;
    pacman) INSTALL_CMD=(pacman -S --needed --noconfirm "${MISSING[@]}") ;;
    zypper) INSTALL_CMD=(zypper install -y "${MISSING[@]}") ;;
    *) INSTALL_CMD=() ;;
  esac
  say "Some programs and libraries Jarvis can use are missing: ${MISSING[*]}"
  if [ "${#INSTALL_CMD[@]}" -gt 0 ]; then
    say "Install them with:"
    printf '    sudo %s\n' "${INSTALL_CMD[*]}"
    if [ "$SYSTEM_DEPS" = 1 ]; then
      if [ "$(id -u)" = 0 ]; then
        say "Running as root: ${INSTALL_CMD[*]}"
        "${INSTALL_CMD[@]}" || warn "the system package install failed; Jarvis still installs, with fewer features"
      elif command -v sudo >/dev/null 2>&1; then
        say "Running with sudo (it may ask for your password): ${INSTALL_CMD[*]}"
        sudo "${INSTALL_CMD[@]}" || warn "the system package install failed; Jarvis still installs, with fewer features"
      else
        warn "sudo is not available; run the command above as root"
      fi
    else
      say "(Not run: pass --system-deps to let this script run it, or copy the line above.)"
    fi
  else
    say "Unknown distribution: install the packages that provide those programs."
  fi
else
  say "System libraries and helper programs: all present."
fi

# ---- 3. Jarvis ------------------------------------------------------------------------
EXTRAS="all"
[ "$MINIMAL" = 1 ] && EXTRAS=""
if [ "$GPU" = 1 ]; then
  if [ -n "$EXTRAS" ]; then EXTRAS="${EXTRAS},gpu"; else EXTRAS="gpu"; fi
fi
if [ -n "$EXTRAS" ]; then
  SPEC="jarvis[${EXTRAS}] @ git+${REPO_URL}@${JARVIS_REF}"
else
  SPEC="jarvis @ git+${REPO_URL}@${JARVIS_REF}"
fi

say "Installing Jarvis ${JARVIS_REF} (re-running this upgrades it)"
uv tool install --python 3.13 --force --reinstall-package jarvis --refresh-package jarvis "$SPEC"

BIN_DIR="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
JARVIS_BIN="${BIN_DIR}/jarvis"
[ -x "$JARVIS_BIN" ] || JARVIS_BIN="$(command -v jarvis || true)"
[ -n "$JARVIS_BIN" ] || die "the install finished but the jarvis command was not found in ${BIN_DIR}"

if ! command -v jarvis >/dev/null 2>&1; then
  warn "${BIN_DIR} is not on your PATH yet. Run 'uv tool update-shell' and open a new terminal."
fi
"$JARVIS_BIN" --version

# ---- 4. first run ----------------------------------------------------------------------
have_tty() { [ -t 1 ] && [ -r /dev/tty ] && (: </dev/tty) 2>/dev/null; }

if [ -z "${JARVIS_NO_SETUP:-}" ] && have_tty; then
  say "Starting first-run setup (Ctrl+C to skip; run 'jarvis --setup' any time)"
  "$JARVIS_BIN" --setup </dev/tty || warn "setup did not finish; run 'jarvis --setup' again"
else
  cat <<EOF

Next steps:
  jarvis --setup     choose a provider and store your API key (free key: https://build.nvidia.com)
  jarvis --doctor    check that everything works
  jarvis             start chatting

Uninstall: uv tool uninstall jarvis   (settings and memory stay; see the README for where they live)
EOF
fi
