This self-contained setup script installs the pre-compiled MLX-Audio engine, downloads the **Fish Audio S2 Pro** model, and creates a global `fish-say` terminal command you can run from any folder on any Mac.

**Step 1: What This Script Does Automatically**

1. **Checks Prerequisites:** Checks if `uv` and `ffmpeg` are installed on the Mac. If they are missing, it installs them automatically via Homebrew.
2. **Installs MLX-Audio:** Installs the pre-compiled `mlx-audio` engine globally so you never hit Xcode Metal compiler errors.
3. **Creates a Global `fish-say` Command:** Places a helper script inside `~/.local/bin/fish-say` so you can type `fish-say "Hello world"` from anywhere in your Terminal.
4. **Pre-Downloads the Model:** Runs a short test sentence to automatically download the **8-bit MLX** model (or **BF16** if you pass `--bf16`) and speaks a confirmation message out loud.

---

**Step 2: The All-in-One Mac Setup Script (`setup-fish-mlx.sh`)**

Save this file as `setup-fish-mlx.sh` (or copy it to a USB drive / GitHub Gist to run on all your Macs):

```bash
#!/usr/bin/env bash
set -e

# Default to the fast 8-bit MLX model (pass --bf16 to the script for the full 16-bit model)
MODEL="mlx-community/fish-audio-s2-pro-8bit"
if [[ "$1" == "--bf16" ]]; then
  MODEL="mlx-community/fish-audio-s2-pro-bf16"
fi

echo "===================================================="
echo " Starting Fish Audio S2 Pro MLX Setup"
echo " Selected Model: $MODEL"
echo "===================================================="

# 1. Remove any leftover broken Swift build folder
if [ -d "$HOME/mlx-audio-swift" ]; then
  echo "[1/5] Cleaning up old mlx-audio-swift build folder..."
  rm -rf "$HOME/mlx-audio-swift"
else
  echo "[1/5] No broken Swift build folders found."
fi

# 2. Check and install uv and ffmpeg via Homebrew if missing
echo "[2/5] Checking audio and package helpers (uv & ffmpeg)..."
if ! command -v brew >/dev/null 2>&1; then
  echo "Error: Homebrew is not installed. Visit https://brew.sh first."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv via Homebrew..."
  brew install uv
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg via Homebrew..."
  brew install ffmpeg
fi

# 3. Install pre-compiled MLX-Audio globally via uv
echo "[3/5] Installing pre-compiled mlx-audio engine..."
uv tool install --force mlx-audio --prerelease=allow
uv tool update-shell

# 4. Create a simple global 'fish-say' command in ~/.local/bin
echo "[4/5] Creating global 'fish-say' command in ~/.local/bin..."
mkdir -p "$HOME/.local/bin"

cat << 'EOF' > "$HOME/.local/bin/fish-say"
#!/usr/bin/env bash
# Usage:
#   fish-say "Text to speak out loud"
#   fish-say "Text to save" output.wav
#   FISH_MODEL="mlx-community/fish-audio-s2-pro-bf16" fish-say "Use BF16 model"

MODEL="${FISH_MODEL:-mlx-community/fish-audio-s2-pro-8bit}"
TEXT="$1"
OUTPUT="$2"

if [ -z "$TEXT" ]; then
  echo "Usage: fish-say \"Your text here\" [optional_output.wav]"
  exit 1
fi

if [ -n "$OUTPUT" ]; then
  "$HOME/.local/bin/mlx_audio.tts.generate" \
    --model "$MODEL" \
    --text "$TEXT" \
    --output "$OUTPUT" \
    --play
else
  "$HOME/.local/bin/mlx_audio.tts.generate" \
    --model "$MODEL" \
    --text "$TEXT" \
    --play
fi
EOF

chmod +x "$HOME/.local/bin/fish-say"

# Ensure ~/.local/bin is active in the current terminal session
export PATH="$HOME/.local/bin:$PATH"

# 5. Pre-download the model and run a live audio check
echo "[5/5] Downloading $MODEL (first run only) and testing audio..."
"$HOME/.local/bin/fish-say" "Fish Audio S 2 Pro M L X is installed and ready on this Mac."

echo "===================================================="
echo " Setup Complete!"
echo " Try running: fish-say \"Hello from Apple Silicon\""
echo "===================================================="

```

---

**Step 3: How to Run It on Any Mac**

Open your Terminal, create the script, make it executable, and run it:

```bash
chmod +x setup-fish-mlx.sh
./setup-fish-mlx.sh

```

*(Tip: If you want a specific Mac to default to the full **BF16** studio model during setup, run `./setup-fish-mlx.sh --bf16` instead.)*

---

**Step 4: How to Use Your New `fish-say` Command**

Once the script finishes, you never have to type the long `mlx_audio.tts.generate` command again. Open any Terminal window and use `fish-say`:

```bash
# 1. Speak text out loud immediately (with emotional tags)
fish-say "All background tasks are finished. [sigh] Time for a break."

# 2. Speak text AND save it to a specific WAV file
fish-say "Saving this voice line for a project." voice_line.wav

# 3. Temporarily switch to the full BF16 model for a studio take
FISH_MODEL="mlx-community/fish-audio-s2-pro-bf16" fish-say "Studio quality sixteen bit audio."

```