<div align="center">
    <h2 align="center">Pixels, Points, Polygons: A Global Dataset and Baseline for Multimodal Building Vectorization</h2>
    <!-- <h3 align="center">Arxiv</h3> -->
    <!-- <h3 align="center"><a href="https://raphaelsulzer.de/">Raphael Sulzer<sup>1,2</sup></a><br></h3> -->
    <h3 align="center">Raphael Sulzer<sup>1,2</sup></a><br></h3>
    <h4 align="center"><sup>1</sup>LuxCarta   <sup>2</sup>Inria</h4>
    <img src="./media/teaser.png" width=100% height=100%>
</div>


<!-- [[Project Webpage]()]    [[Paper](https://arxiv.org/abs/2412.07899)]    [[Video]()] -->

## Abstract:

asd

## Highlights

- A global, multimodal dataset of aerial images, aerial lidar point clouds and building polygons
- A library for training and evaluating SOTA deep learning methods on the dataset


## Datasets download and preparation

#TODO
<!-- See [datasets preprocessing](data_preprocess) for instructions on preparing the various datasets for training/inference. -->


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

## Configuration


The project supports hydra configuration which allows to modify any parameter from the command line.
To view all available options run
```
python train.py --help
```


## Training

Start training with the following command:

```
torchrun --nproc_per_node=<num GPUs> train.py model=<pix2poly,hisup,ffl> use_lidar=<true,false> use_images=<true,false> model.batch_size=<batch size> ...

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
