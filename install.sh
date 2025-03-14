#!/bin/bash
set -e

# Local variables
ENV_NAME=ppp
PYTHON=3.11.11

# Installation script for Miniconda3 environments
echo "____________ Pick Miniconda Install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath ~/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath /opt/miniconda3`
fi

if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
    CONDA_DIR=$(conda info | grep 'base environment' | awk '{print $4}')
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide you conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh
echo
echo

echo "________________ Install Conda Environment _______________"
echo

# Check if the environment exists
if conda env list | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
    read -p "Conda environment '$ENV_NAME' already exists. Do you want to remove and reinstall it? (yes/no): " answer

    if [[ "$answer" == "yes" || "$answer" == "y" ]]; then
        # Remove the environment
        conda env remove --name "$ENV_NAME" --yes > /dev/null 2>&1

        # Double-check removal
        if conda env list | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
            echo "Failed to remove the environment '$ENV_NAME'."
            exit 1
        else
            echo "Conda environment '$ENV_NAME' removed successfully."
        fi

        ## Create a conda environment
        echo "Create conda environment '$ENV_NAME'."
        conda create -y --name $ENV_NAME python=$PYTHON > /dev/null 2>&1

    elif [[ "$answer" == "no" || "$answer" == "n" ]]; then
        echo "Installing in existing environment..."
    else
        echo "Invalid input. Please enter yes or no."
    fi
else
  ## Create a conda environment
  echo "Create conda environment '$ENV_NAME'."
  conda create -y --name $ENV_NAME python=$PYTHON > /dev/null 2>&1
fi


# Activate the env
echo "Activating ${ENV_NAME} conda environment."
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}


echo "________________ Install Required Packages _______________"
echo

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers==4.32
# pip install pycocotools
# pip install torchmetrics
# pip install wandb
# pip install timm
# pip install matplotlib
# pip install albumentations
# pip install shapely
# pip install hydra-core

# # # problem with torch:tms? do this:
# # # https://github.com/huggingface/diffusers/issues/8958#issuecomment-2253055261

# # ## for inria_to_coco.py
# # conda install conda-forge::imagecodecs -y

# # ## for lidar_poly_dataloader
# # conda install conda-forge::gcc_linux-64=10 conda-forge::gxx_linux-64=10 -y # otherwise copclib install bugs
# pip install laspy[laszip]
# pip install colorlog
# pip install descartes==1.1.0
# pip install scikit-image
# pip3 install -U scikit-learn

pip install -e .