import os
from huggingface_hub import snapshot_download

script_dir = os.path.dirname(os.path.abspath(__file__))
local_dir = os.path.join(script_dir, "../dataset/custom_droid_dataset")

snapshot_download(
    repo_id="mousecpn/custom_droid_dataset",
    repo_type="dataset",
    local_dir=local_dir,
    token=None,
    resume_download=True
)