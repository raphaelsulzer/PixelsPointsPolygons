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

def upload_single_file(local_path,repo_path=None):
    try:
        # repo_path = os.path.relpath(local_path, LOCAL_DIR)
        logger.info(f"Uploading {repo_path}...")

        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
        )
        return (repo_path, "success")
    except HfHubHTTPError as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return (local_path, "failed")

def upload_hf_large_folder(local_path):
    
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=local_path,
    )

def upload_hf_folder(local_path,repo_path):
    
    logger.info(f"Uploading folder {local_path} to {repo_path}...")
    
    try:
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token
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
    
    ################ UPLOAD LARGE FOLDER ################   
    # upload_hf_large_folder(LOCAL_DIR)
    ################ UPLOAD LARGE FOLDER ################   


    ################ UPLOAD FOLDER ################   
    splits = ["train", "val", "test"]
    countries = ["NY","NZ"]
    modalities = ["images","lidar"]
    
    countries = ["NZ"]
    # modalities = ["images"]
    # splits = ["test"]
    
    for split in splits:
        
        for country in countries:
            
            for modality in modalities:
        
                all_folders = f"/data/rsulzer/lidarpoly/224/{modality}/{split}/{country}"
                
                sub_folders = os.listdir(all_folders)
                
                for folder in sub_folders:
                    in_folder = os.path.join(all_folders, folder)
                    
                    assert os.path.isdir(in_folder), f"Expected {in_folder} to be a directory."
                
                    repo_folder = f"data/224/{modality}/{split}/{country}/{folder}"
                    
                    try:
                        upload_hf_folder(in_folder, repo_folder)
                    except Exception as e:
                        logger.error(f"Failed to upload folder {in_folder}: {e}")
                        continue
    ############### UPLOAD FOLDER ################
          
    
    ################ DELETE FILES ################
    # repo_file = "data/224/images/test/NZ/0/asd"
    # delete_hf_file(repo_file)
    ################ DELETE FILES ################

    ################ DELETE FOLDER ################
    # repo_folder = "data"
    # delete_hf_folder(repo_folder)
    ################ DELETE FOLDER ################

    # ################ UPLOAD FILES ################
    # all_files = list(list_files_recursively(LOCAL_DIR))
    # logger.info(f"Found {len(all_files)} files to upload.")

    # with Pool(NUM_WORKERS) as pool:
    #     results = pool.map(upload_single_file, all_files)

    # # Summary
    # success = sum(1 for _, status in results if status == "success")
    # failed = sum(1 for _, status in results if status == "failed")
    # logger.info(f"Upload complete. ✅ {success} succeeded, ❌ {failed} failed.")
    # ################ UPLOAD FILES ################
    
    # file = "/data/rsulzer/PixelsPointsPolygons/data/224/images/test/NZ/0/image4753_NZ_test.tif"
    # repo_file = "data/224/images/test/NZ/0/"
    # upload_single_file(file,repo_file)
