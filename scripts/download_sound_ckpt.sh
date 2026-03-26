#!/bin/bash
set -e

MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
MODEL_DIR="$(dirname "$0")/../ckpts"
ZIP_FILE="$MODEL_DIR/vosk-model-en-us-0.22.zip"

mkdir -p "$MODEL_DIR"

echo "Downloading Vosk model..."
curl -L "$MODEL_URL" -o "$ZIP_FILE"

echo "Unzipping model..."
unzip -o "$ZIP_FILE" -d "$MODEL_DIR"

echo "Cleaning up zip file..."
rm "$ZIP_FILE"

echo "Done. Model is in $MODEL_DIR."
