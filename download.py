from huggingface_hub import snapshot_download
import os

# 1. The exact name of the model on Hugging Face
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# 2. Choose where you want to save it. 
# If you leave this as None, it saves to your default C: drive cache.
# If you want it on your G: drive (since I saw that in your terminal), set the path!
CUSTOM_DIR = "google/gemma-2-9b-it"

print(f"Starting download for {MODEL_ID}...")
print("This is about 14GB, so grab a coffee! ☕\n")

# 3. Download the model
try:
    if CUSTOM_DIR:
        # Create the folder if it doesn't exist
        os.makedirs(CUSTOM_DIR, exist_ok=True)
        
        # Download straight to your custom folder
        download_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=CUSTOM_DIR,
            local_dir_use_symlinks=False # Forces actual file download instead of shortcuts
        )
        print(f"\n✅ SUCCESS! Model downloaded completely to: {download_path}")
    else:
        # Download to the default Hugging Face cache folder
        download_path = snapshot_download(repo_id=MODEL_ID)
        print(f"\n✅ SUCCESS! Model cached to default huggingface directory.")

except Exception as e:
    print(f"\n❌ Download failed: {e}")