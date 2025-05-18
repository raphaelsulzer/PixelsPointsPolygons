import os
import logging
from multiprocessing import Pool, cpu_count
from huggingface_hub import HfApi, HfFolder, upload_file
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi
api = HfApi()


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


def upload_hf_folder(local_path,repo_path):
    try:
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        logger.info(f"Successfully uploaded folder {local_path}.")
    except HfHubHTTPError as e:
        logger.error(f"Failed to upload folder {local_path}: {e}")
        
def delete_hf_folder(repo_path):
    try:
        api.delete_folder(
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        logger.info(f"Successfully deleted {repo_path}.")
    except HfHubHTTPError as e:
        logger.error(f"Failed to delete {repo_path}: {e}")


        
def delete_hf_file(repo_file):
    try:
        api.delete_file(
            path_in_repo=repo_file,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        logger.info(f"Successfully deleted {repo_file}.")
    except HfHubHTTPError as e:
        logger.error(f"Failed to delete {repo_file}: {e}")

if __name__ == "__main__":
    
    
    # splits = ["train", "val", "test"]
    # countries = ["CH","NY","NZ"]
    # modalities = ["images","lidar"]
    
    # countries = ["NY"]
    # modalities = ["images"]
    
    # # splits = ["val"]
    # for split in splits:
        
    #     for country in countries:
            
    #         for modality in modalities:
        
    #             all_folders = f"/data/rsulzer/lidarpoly/224/{modality}/{split}/{country}"
                
    #             sub_folders = os.listdir(all_folders)
                
    #             for folder in sub_folders:
    #                 in_folder = os.path.join(all_folders, folder)
                    
    #                 assert os.path.isdir(in_folder), f"Expected {in_folder} to be a directory."
                
    #                 repo_folder = f"data/224/{modality}/{split}/{country}/{folder}"
                    
    #                 try:
    #                     upload_hf_folder(in_folder, repo_folder)
    #                 except Exception as e:
    #                     logger.error(f"Failed to upload folder {in_folder}: {e}")
    #                     continue
                    
    
    ################ DELETE FILES ################
    # repo_file = "lidar481_Switzerland_val.copc.laz"
    # delete_hf_file(repo_file)
    
    # repo_folder = "data/224/images/{split}"
    # delete_hf_folder(repo_folder)
    ################ DELETE FILES ################

    all_files = list(list_files_recursively(LOCAL_DIR))
    logger.info(f"Found {len(all_files)} files to upload.")

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(upload_single_file, all_files)

    # Summary
    success = sum(1 for _, status in results if status == "success")
    failed = sum(1 for _, status in results if status == "failed")
    logger.info(f"Upload complete. ✅ {success} succeeded, ❌ {failed} failed.")
