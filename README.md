<div align="center">
    <h2 align="center">Pixels, Points, Polygons: A Global Dataset and Baseline for Multimodal Building Vectorization</h2>
    <h3 align="center">Arxiv</h3>
    <a href="https://raphaelsulzer.de/">Raphael Sulzer<sup>1,2</sup></a><br>
    <sup>1</sup>LuxCarta <sup>2</sup>Inria
    <!-- <img src="./assets/sfo7.png" width=80% height=80%> -->
</div>


<!-- [[Project Webpage]()]    [[Paper](https://arxiv.org/abs/2412.07899)]    [[Video]()] -->

### Abstract:

asd

## Installation

To create a conda environment named `ppp` and install as a python package with all dependencies run
```
bash install.sh
```

or, if you want to manage the environment yourself run
```
pip install -r requirements-torch-cuda.txt
pip install .
```


## Datasets download and preparation

See [datasets preprocessing](data_preprocess) for instructions on preparing the various datasets for training/inference.

## Configurations


The project uses hydra-conf which allows to modify any parameter from the command line.
To view all available options run
```
python train.py --help
```


## Training

Start training with the following command:

```
torchrun --nproc_per_node=<num GPUs> train.py model=<pix2poly,hisup,ffl> use_lidar=true use_images=true model.batch_size=<batch size> multi_gpu=true ...

```

## Prediction

```
torchrun --nproc_per_node=<num GPUs> predict.py model=<pix2poly,hisup,ffl> checkpoint=validation_best ...

```

## Evaluation

```
python evaluate.py model=<pix2poly,hisup,ffl> checkpoint=validation_best
```


## Citation

If you find our work useful, please consider citing:
```bibtex
...
```

## Acknowledgements

This repository benefits from the following open-source work. We thank the authors for their great work.

1. [Frame Field Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning)
2. [HiSup](https://github.com/SarahwXU/HiSup)
3. [Pix2Poly](https://github.com/yeshwanth95/Pix2Poly)
