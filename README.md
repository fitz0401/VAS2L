# Install
```
# Env
conda create -n vas2l python=3.10 -y
conda activate vas2l
pip install -e .
```

## Sound2Launage Module
```
pip install vosk sounddevice
apt-get install libportaudio2
bash scripts/download_sound_ckpy.sh
# choose the model you need from: https://alphacephei.com/vosk/models
```

## Vision-Action2Language Module
```
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
--index-url https://download.pytorch.org/whl/cu118
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.8/

# Semantic-SAM
pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
cd vision_module/ops && bash make.sh && cd ../..

# GroundedSAM
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO

# VLMs
pip install transformers==4.57.0 
pip install 'accelerate>=0.26.0'
pip install timm==0.9.12

# Tips: `hf auth login` for using --model gemma
```

## Dataset
Ref: https://droid-dataset.github.io/droid/the-droid-dataset.html
```


```