<div align="center">
    <h2 align="center">Pixels, Points, Polygons: A Global Dataset and Baseline for Multimodal Building Vectorization</h2>
    <h3 align="center">Arxiv</h3>
    <a href="https://raphaelsulzer.de/">Raphael Sulzer<sup>1</sup></a><br>
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
