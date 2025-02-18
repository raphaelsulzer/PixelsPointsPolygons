<div align="center">
    <h2 align="center">Pix2Poly: A Sequence Prediction Method for End-to-end Polygonal Building Footprint Extraction</h2>
    <h3 align="center">WACV 2025</h3>
    <a href="https://yeshwanth95.github.io/">Yeshwanth Kumar Adimoolam<sup>1</sup></a>, <a href="https://poullis.org/">Charalambos Poullis<sup>2</sup></a>, <a href="https://melinos.github.io/">Melinos Averkiou<sup>1</sup></a><br>
    <sup>1</sup>CYENS CoE, Cyprus, <sup>2</sup>Concordia University
    <img src="./assets/sfo7.png" width=80% height=80%>
</div>


[[Project Webpage]()]    [[Paper](https://arxiv.org/abs/2412.07899)]    [[Video]()]

### Abstract:

Extraction of building footprint polygons from remotely sensed data is essential for several urban understanding tasks such as reconstruction, navigation, and mapping. Despite significant progress in the area, extracting accurate polygonal building footprints remains an open problem. In this paper, we introduce Pix2Poly, an attention-based end-to-end trainable and differentiable deep neural network capable of directly generating explicit high-quality building footprints in a ring graph format. Pix2Poly employs a generative encoder-decoder transformer to produce a sequence of graph vertex tokens whose connectivity information is learned by an optimal matching network. Compared to previous graph learning methods, ours is a truly end-to-end trainable approach that extracts high-quality building footprints and road networks without requiring complicated, computationally intensive raster loss functions and intricate training pipelines. Upon evaluating Pix2Poly on several complex and challenging datasets, we report that Pix2Poly outperforms state-of-the-art methods in several vector shape quality metrics while being an entirely explicit method.

### Method

<div align="center">
    <img src="./assets/pix2poly_overall_bg_white.png" width=40% height=40%>
    <!--<div align="center">-->
    <!--    <img src="./assets/vertex_sequence_detector.png" width=20% height=20%>-->
    <!--    <img src="./assets/optimal_matching_network.png" width=20% height=20%>-->
    <!--</div>-->
</div>

__Overview of the Pix2Poly architecture:__ The Pix2Poly architecture consists of three major components: (i) The Discrete Sequence Tokenizer, (ii) the Vertex Sequence Detector, (iii) and the Optimal Matching Network. The Discrete Sequence Tokenizer is used to convert the continuous building corner coordinates into discrete building corner coordinate tokens which form the ground truth for training Pix2Poly. The Vertex Sequence Detector is an encoder-decoder transformer that predicts a sequence of sequence of discrete building corner coordinate tokens. The Optimal Matching Network takes the predicted corner coordinate tokens and the per-corner features from the vertex sequence detector and predicts a N X N permutation matrix which contains the connectivity information between every possible corner pair. Together, the predicted building corners and permutation matrix are used to recover the final building polygons.

## Installation

Pix2Poly was developed with `python=3.11`, `pytorch=2.1.2`, `pytorch-cuda=11.8`, `timm=0.9.12`, `transformers=4.32.1`

Create a conda environment with the following specification:

```
Conda requirements:
channels:
  - defaults
dependencies:
  - torchvision=0.16.2
  - pytorch=2.1.2
  - pytorch-cuda=11.8
  - torchaudio=2.1.2
  - timm=0.9.12
  - transformers=4.32.1
  - pycocotools=2.0.6
  - torchmetrics=1.2.1
  - tensorboard=2.15.1
  - pip:
    - albumentations==1.3.1
    - imageio==2.33.1
    - matplotlib-inline==0.1.6
    - opencv-python-headless==4.8.1.78
    - scikit-image==0.22.0
    - scikit-learn==1.3.2
    - scipy==1.11.4
    - shapely==2.0.4
```


## Datasets preparation

See [datasets preprocessing](datasets) for instructions on preparing the various datasets for training/inference.

## Configurations

## Training

Start training with the following command:

```
torchrun --nproc_per_node=<num GPUs> train_ddp.py 
```

## Prediction

### (i) INRIA Dataset
To generate predictions for the INRIA dataset, run the following:
```shell
python predict_inria_coco_val_set.py -d inria_coco_224_negAug \
                                     -e <path_to_exp_folder> \
                                     -c <checkpoint_name> \
                                     -o <Name of output subdirectory>
python postprocess_coco_parts.py  # change input and output paths in L006 to L010.
```

### (ii) Spacenet 2 Dataset
To generate predictions for the Spacenet 2 dataset, run the following:
```shell
python predict_spacenet_coco_val_set.py -d spacenet_coco \
                                        -e <path_to_exp_folder> \
                                        -c <checkpoint_name> \
                                        -o <Name of output subdirectory>
python postprocess_coco_parts.py  # change input and output paths in L006 to L010.
```

### (iii) WHU Buildings Dataset
To generate predictions for the WHU Buildings dataset, run the following:
```shell
python predict_whu_buildings_coco_test_set.py -d whu_buildings_224_coco \
                                              -e <path_to_exp_folder> \
                                              -c <checkpoint_name> \
                                              -o <Name of output subdirectory>
python postprocess_coco_parts.py  # change input and output paths in L006 to L010.
```
### (iv) Massachusetts Roads Dataset
To generate predictions for the Massachusetts Roads dataset, run the following:
```shell
python predict_mass_roads_test_set.py -e <path_to_exp_folder> \
-c <checkpoint_name> \
-s <split> \  # 'test' or 'val'
--img_size 224 \
--input_size 224 \
--batch_size 24 \  # modify according to resources
```

## Evaluation

Once predictions are made, metrics can be computed for the predicted files as follows:

```bash
python evaluation.py --gt-file path/to/ground/truth/annotation.json --dt-file path/to/prediction.json --eval-type <metric_type>
```

where `metric_type` can be one of the following: `ciou`, `angle`, `polis`, `topdig`.

## Citation

If you find our work useful, please consider citing:
```bibtex
@misc{adimoolam2024pix2poly,
      title={Pix2Poly: A Sequence Prediction Method for End-to-end Polygonal Building Footprint Extraction from Remote Sensing Imagery},
      author={Yeshwanth Kumar Adimoolam and Charalambos Poullis and Melinos Averkiou},
      year={2024},
      eprint={2412.07899},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07899},
}
```

## Acknowledgements

This repository benefits from the following open-source work. We thank the authors for their great work.

1. [Pix2Seq - official repo](https://github.com/google-research/pix2seq)
2. [Pix2Seq - unofficial repo](https://github.com/moein-shariatnia/Pix2Seq)
3. [Frame Field Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning)
4. [PolyWorld](https://github.com/zorzi-s/PolyWorldPretrainedNetwork)
5. [HiSup](https://github.com/SarahwXU/HiSup)
