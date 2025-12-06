This repository provides code and benchmarks for evaluating Test-Time Adaptation (TTA) algorithms under realistic conditions, including unsupervised hyperparameter selection.


## Citation
```bibtex
@inproceedings{cygert2026realistic,
  author    = {Sebastian Cygert and Damian S{\'o}jka and Tomasz Trzci{\'n}ski and Bart{\l}omiej Twardowski},
  title     = {Realistic Evaluation of Test-Time Adaptation Algorithms: Unsupervised Hyperparameter Selection},
  booktitle = {Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP)},
  year      = {2026},
  address   = {Marbella, Spain},
}
```

## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate tta
``` 

## Classification
This repository contains an extensive collection of different methods, datasets and settings,
which we evaluate in a comprehensive benchmark. 

<details open>
<summary>Features</summary>

This repository allows to study a wide range of different datasets, models, settings, and methods. A quick overview is given below:

- **Datasets**
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
  - `imagenet_r` [ImageNet-R](https://github.com/hendrycks/imagenet-r)
  - `imagenet_v2` [ImageNet-V2](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
  
- **Models**
  - For adapting to ImageNet variations, all pre-trained models available in [Torchvision](https://pytorch.org/vision/0.14/models.html) or [timm](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13) can be used.
  - For the corruption benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench) can be used.
  - For the DomainNet-126 benchmark, there is a pre-trained model for each domain.
 
- **Settings**
  - `continual` Train the model on a sequence of domains without knowing when a domain shift occurs.
  - `correlated` Same as the continual setting but the samples of each domain are further sorted by class label.

- **Methods**
  - The repository currently supports the following methods: BN-0 (source), [TENT](https://openreview.net/pdf?id=uXl3bZLkr3c),
  [MEMO](https://openreview.net/pdf?id=vn74m_tWu8O), [EATA](https://arxiv.org/abs/2204.02610),
  [AdaContrast](https://arxiv.org/abs/2204.10377), [LAME](https://arxiv.org/abs/2201.05718), 
  [SAR](https://arxiv.org/abs/2302.12400), [RMT](https://arxiv.org/abs/2211.13081).

</details>

### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.
- *CIFAR10-to-CIFAR10-C*: the data is automatically downloaded.
- *CIFAR100-to-CIFAR100-C*: the data is automatically downloaded.
- *ImageNet-to-ImageNet-C*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF).s).
- *ImageNet-to-ImageNet-R*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-R](https://github.com/hendrycks/imagenet-r).
- *ImageNet-to-ImageNet-V2*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-V2](https://huggingface.co/datasets/vaishaal/Image
- *DomainNet-126*: download the 6 splits of the [cleaned version](http://ai.bu.edu/M3SDA/). Following [MME](https://arxiv.org/abs/1904.06487), DomainNet-126 only uses a subset that contains 126 classes from 4 domains.

Next, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. For the individual datasets, the directory names are specified in `conf.py` as a dictionary (see function `complete_data_dir_path`). In case your directory names deviate from the ones specified in the mapping dictionary, you can simply modify them.

### Run Experiments

We provide config and job files (subdirectory `jobs`) for all experiments and methods. Simply run the following Python file with the corresponding config file.

Example run:
```bash
python test_time.py --cfg cfgs/cifar100_c/tent.yaml
```

### Acknowledgements
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)
+ SAR [official](https://github.com/mr-eggplant/SAR)

# CREDITS
This open source benchmark is based on the official repository for the following works:
- [Introducing Intermediate Domains for Effective Self-Training during Test-Time](https://arxiv.org/abs/2208.07736)
- [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://arxiv.org/abs/2211.13081) (CVPR2023)
- [Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction](https://arxiv.org/abs/2306.00650) (WACV2024)
