<div align="center">
    <h2 align="center">Pixels, Points, Polygons: A Global Dataset and Baseline for Multimodal Building Vectorization</h2>
    <h3 align="center">Arxiv</h3>
    <a href="https://raphaelsulzer.de/">Raphael Sulzer<sup>1,2</sup></a><br>
    <sup>1</sup>LuxCarta <sup>2</sup>Inria
    <!-- <img src="./assets/sfo7.png" width=80% height=80%> -->
</div>


[[Project Webpage]()]    [[Paper](https://arxiv.org/abs/2412.07899)]    [[Video]()]

### Abstract:

asd

## Installation

`bash install.sh`

## Datasets preparation

See [datasets preprocessing](data_preprocess) for instructions on preparing the various datasets for training/inference.

## Configurations

## Training

Start training with the following command:

```
torchrun --nproc_per_node=<num GPUs> train_ddp.py 
```

## Prediction



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
