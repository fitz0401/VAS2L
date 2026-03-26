#!/bin/bash
set -e

CKPT_DIR="$(dirname "$0")/../ckpts"
mkdir -p "$CKPT_DIR"

echo "Downloading swinl_only_sam_many2many.pth..."
wget -O "$CKPT_DIR/swinl_only_sam_many2many.pth" https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth

echo "Downloading sam_vit_h_4b8939.pth..."
wget -O "$CKPT_DIR/sam_vit_h_4b8939.pth" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "Downloading groundingdino_swint_ogc.pth..."
wget -O "$CKPT_DIR/groundingdino_swint_ogc.pth" https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

echo "All vision checkpoints downloaded to $CKPT_DIR."