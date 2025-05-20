<div align="center">
    <h1 align="center">The P<sup>3</sup> dataset: Pixels, Points and Polygons <br> for Multimodal Building Vectorization</h1>
    <h3><align="center">Raphael Sulzer<sup>1,2</sup> &nbsp;&nbsp;&nbsp; Liuyun Duan<sup>1</sup>
    &nbsp;&nbsp;&nbsp; Nicolas Girard<sup>1</sup>&nbsp;&nbsp;&nbsp; Florent Lafarge<sup>2</sup></a></h3>
    <align="center"><sup>1</sup>LuxCarta Technology <br>  <sup>2</sup>Centre Inria d'Université Côte d'Azur
    <img src="./teaser.jpg" width=100% height=100%>
    <b>Figure 1</b>: A view of our dataset of Zurich, Switzerland
</div>

## Abstract

<div align="justify">
We present the P<sup>3</sup> dataset, a large-scale multimodal benchmark for building vectorization, constructed from aerial LiDAR point clouds, high-resolution aerial imagery, and vectorized 2D building outlines, collected across three continents. The dataset contains over 10 billion LiDAR points with decimeter-level accuracy and RGB images at a ground sampling distance of 25 cm. While many existing datasets primarily focus on the image modality, P<sup>3</sup> offers a complementary perspective by also incorporating dense 3D information. We demonstrate that LiDAR point clouds serve as a robust modality for predicting building polygons, both in hybrid and end-to-end learning frameworks. Moreover, fusing aerial LiDAR and imagery further improves accuracy and geometric quality of predicted polygons. The P<sup>3</sup> dataset is publicly available, along with code and pretrained weights of three state-of-the-art models for building polygon prediction at https://github.com/raphaelsulzer/PixelsPointsPolygons.
</div>

## Highlights

- A global, multimodal dataset of aerial images, aerial LiDAR point clouds and building outline polygons, available at [huggingface.co/datasets/rsi/PixelsPointsPolygons](https://huggingface.co/datasets/rsi/PixelsPointsPolygons) 
- A library for training and evaluating state-of-the-art deep learning methods on the dataset, available at [github.com/raphaelsulzer/PixelsPointsPolygons](https://github.com/raphaelsulzer/PixelsPointsPolygons)
- Pretrained model weights, available at [huggingface.co/rsi/PixelsPointsPolygons](https://huggingface.co/rsi/PixelsPointsPolygons) 


## Dataset

### Download

You can download the dataset at [huggingface.co/datasets/rsi/PixelsPointsPolygons](https://huggingface.co/datasets/rsi/PixelsPointsPolygons) .




### Overview

<div align="left">
    <img src="./worldmap.jpg" width=60% height=50%>
</div>


<!-- ### Prepare custom tile size

See [datasets preprocessing](data_preprocess) for instructions on preparing a dataset with different tile sizes. -->


## Code 

### Download

```
git clone https://github.com/raphaelsulzer/PixelsPointsPolygons
```

### Requirements

To create a conda environment named `p3` and install the repository as a python package with all dependencies run
```
bash install.sh
```

or, if you want to manage the environment yourself run
```
pip install -r requirements-torch-cuda.txt
pip install .
```
⚠️ **Warning**: The implementation of the LiDAR point cloud encoder uses Open3D-ML. Currently, Open3D-ML officially only supports the PyTorch version specified in `requirements-torch-cuda.txt`.



<!-- ## Model Zoo


| Model                     | \<model>  | Encoder                   | \<encoder>            |Image  |LiDAR  | IoU       | C-IoU     |
|---------------            |----       |---------------            |---------------        |---    |---    |-----      |-----       |
| Frame Field Learning      |\<ffl>     | Vision Transformer (ViT)  | \<vit_cnn>            | ✅    |       | 0.85      | 0.90      |
| Frame Field Learning      |\<ffl>     | PointPillars (PP) + ViT   | \<pp_vit_cnn>         |       | ✅    | 0.80      | 0.88      |
| Frame Field Learning      |\<ffl>     | PP+ViT \& ViT             | \<fusion_vit_cnn>     | ✅    |✅     | 0.78      | 0.85      |
| HiSup                     |\<hisup>   | Vision Transformer (ViT)  | \<vit_cnn>            | ✅    |       | 0.85      | 0.90      |
| HiSup                     |\<hisup>   | PointPillars (PP) + ViT   | \<pp_vit_cnn>         |       | ✅    | 0.80      | 0.88      |
| HiSup                     |\<hisup>   | PP+ViT \& ViT             | \<fusion_vit>         | ✅    |✅     | 0.78      | 0.85      |
| Pix2Poly                  |\<pix2poly>| Vision Transformer (ViT)  | \<vit>                | ✅    |       | 0.85      | 0.90      |
| Pix2Poly                  |\<pix2poly>| PointPillars (PP) + ViT   | \<pp_vit>             |       | ✅    | 0.80      | 0.88      |
| Pix2Poly                  |\<pix2poly>| PP+ViT \& ViT             | \<fusion_vit>         | ✅    |✅     | 0.78      | 0.85      | -->

### Configuration

The project supports hydra configurations which allow to modify any parameter from the command line, such as the model and encoder types from the table above.
To view all available options run
```
python scripts/train.py --help
```

<!-- The most important parameters are described below:
<details>
<summary>CLI Parameters</summary>

- **experiment:** Path to an experiment.yaml file
- **model:** Overwrite for a specific model

</details> -->

### Training

Start training with the following command:

```
torchrun --nproc_per_node=<num GPUs> scripts/train.py model=<model> encoder=<encoder> model.batch_size=<batch size> ...

```

### Prediction

```
torchrun --nproc_per_node=<num GPUs> scripts/predict.py model=<model> checkpoint=best_val_iou ...

```

### Evaluation

```
python scripts/evaluate.py model=<model> checkpoint=best_val_iou
```
<!-- ## Trained models

asd -->


<!-- ## Results

#TODO Put paper main results table here -->


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