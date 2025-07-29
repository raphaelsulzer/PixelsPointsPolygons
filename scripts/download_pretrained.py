import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download pretrained models to a specified local directory.")
    parser.add_argument(
        "--model-root",
        type=str,
        required=True,
        help="Path to the directory where the model should be downloaded"
    )
    args = parser.parse_args()

    model_root = args.model_root

    print(f"Downloading model to: {model_root}")
    snapshot_download(
        repo_id="rsi/PixelsPointsPolygons",  # replace with your actual model repo
        repo_type="model",
        local_dir=model_root,
        local_dir_use_symlinks=False
    )
    print(f"✅ Pretrained models successfully downloaded to: {model_root}")

if __name__ == "__main__":
    main()
