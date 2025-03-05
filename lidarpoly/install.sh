conda create -n p2p python=3.11

conda install conda-forge::gcc_linux-64=10 conda-forge::gxx_linux-64=10 -y # otherwise copclib install bugs
pip install copclib
conda install conda-forge::colorlog -y

# for the metrics submodule
conda install conda-forge::descartes=1.1.0 -y