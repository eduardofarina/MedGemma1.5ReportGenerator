"""Download MedGemma model with resume support."""
import os

# Disable symlinks warning and use file copying instead
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

token = os.getenv("HF_TOKEN")
print(f"Token loaded: {'Yes' if token else 'No'}")

print("Downloading google/medgemma-1.5-4b-it...")
# Use local_dir to avoid symlink issues on Windows
local_dir = os.path.join(os.path.dirname(__file__), "models", "medgemma-1.5-4b-it")
os.makedirs(local_dir, exist_ok=True)

path = snapshot_download(
    "google/medgemma-1.5-4b-it",
    token=token,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Model downloaded to: {path}")
