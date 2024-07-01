import shutil
import os

hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
torch_cache_dir = os.path.expanduser("~/.cache/torch/pyannote")

shutil.rmtree(hf_cache_dir, ignore_errors=True)
shutil.rmtree(torch_cache_dir, ignore_errors=True)
