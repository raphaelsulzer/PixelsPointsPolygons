import os
import logging
from multiprocessing import Pool, cpu_count
from huggingface_hub import HfApi, HfFolder, upload_file
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import upload_large_folder

# --- Configuration ---
REPO_ID = "rsi/PixelsPointsPolygons"   # Change this
LOCAL_DIR = "/data/rsulzer/PixelsPointsPolygons"        # Change this
REPO_TYPE = "dataset"
NUM_WORKERS = min(1, cpu_count())        # Tune as needed

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hf-uploader")

# --- Prepare API ---
api = HfApi()
token = HfFolder.get_token()
assert token, "You must be logged in. Run `huggingface-cli login` first."

def list_files_recursively(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.startswith('.'):  # Skip hidden files
                continue
            yield os.path.join(root, f)

def upload_single_file(local_path):
    try:
        relative_path = os.path.relpath(local_path, LOCAL_DIR)
        logger.info(f"Uploading {relative_path}...")

        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=relative_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
        )
        return (relative_path, "success")
    except HfHubHTTPError as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return (local_path, "failed")




if __name__ == "__main__":
    
    # upload_large_folder(
    #     folder_path="/data/rsulzer/PixelsPointsPolygons",  # current directory (or use a temp dir if needed)
    #     repo_id=REPO_ID,
    #     repo_type=REPO_TYPE,
    #     allow_patterns=["README.md"],  # only upload this file
    #     # commit_message="Update README",
    # )
    
    all_files = list(list_files_recursively(LOCAL_DIR))
    logger.info(f"Found {len(all_files)} files to upload.")

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(upload_single_file, all_files)

    # Summary
    success = sum(1 for _, status in results if status == "success")
    failed = sum(1 for _, status in results if status == "failed")
    logger.info(f"Upload complete. ✅ {success} succeeded, ❌ {failed} failed.")
