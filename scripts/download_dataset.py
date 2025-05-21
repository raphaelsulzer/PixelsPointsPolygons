import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download the P3 dataset to a specified local directory.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the directory where the dataset should be downloaded"
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root

    print(f"Downloading dataset to: {dataset_root}")
    snapshot_download(
        repo_id="rsi/PixelsPointsPolygons",
        repo_type="dataset",
        local_dir=dataset_root,
        local_dir_use_symlinks=False
    )
    print(f"✅ Dataset successfully downloaded to: {dataset_root}")

if __name__ == "__main__":
    main()
