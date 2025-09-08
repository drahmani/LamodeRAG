from huggingface_hub import snapshot_download
import os
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
local_dir = os.path.expanduser("~/LLM/mistral/mistral-7b-instruct")
# Downloads all model files to cache
snapshot_download(model_name, local_dir)

