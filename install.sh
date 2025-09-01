#!/bin/bash
set -e

# Local variables
ENV_NAME=p3
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
  echo "Create conda environment '$ENV_NAME'..."
  conda create -y --name $ENV_NAME python=$PYTHON > /dev/null 2>&1
fi


# Activate the env
echo "Activating ${ENV_NAME} conda environment."
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}


echo "________________ Install Required Packages _______________"
echo

export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"

# install a cudatoolkit 12.1 to match the version specified in requirements-torch-cuda.txt
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit=12.1.1 -y
pip install -r requirements-torch-cuda.txt

# conda install -c nvidia/label/cuda-11.7.1 cuda-toolkit=11.7.1 -y
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

pip install -e .

echo "________________ Install Pytorch_Lydorn and Lydorn_Utils for FFL _______________"
cd ./ffl_submodules/lydorn_utils
pip install -e .
cd ../pytorch_lydorn
pip install -e .
cd ../..

### take this out of the script because it doesn't work on the front end of g5k and probably jz too, because cuda is not properly installed there
### instead run this inside the job submission script
## make the afm module for hisup
echo "________________ Install AFM module for HiSup _______________"
cd ./pixelspointspolygons/models/hisup/afm_module
make

echo "________________ Installation Completed Successfully _______________"
echo "Run 'conda activate ${ENV_NAME}' to activate the environment."
