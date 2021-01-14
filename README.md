# CPM: Color-Pattern Makeup Transfer

| ![teaser.png](./imgs/teaser.png) | 
|:--:| 
| CPM: Color-Pattern Makeup Transfer, implementation for *[Lipstick ain't enough: Beyond Color-Matching for In-the-Wild Makeup Transfer]()* Link will be here very soon |

---

##### Table of Content

1. [Getting Started](#getting-started)
	- [Requirements](#requirements)
	- [Quick Start (Usage)](#usage)
1. [About Data](#about-data)
1. [Training/ Test](#train-&-test)
1. [Common Issues](#common-issues)

---

### Getting Started

##### Requirements

- Install packages

`conda env create -f environment.yml`

- Download pretrained models:

```
mkdir checkpoints
wget color.pth
wget pattern.pth
```

- Download PRNet 

##### Usage

- Color+Pattern: `python main.py --style ./imgs/style-1.png --input ./imgs/non-makeup.png`
- Color Only: `python main.py --style ./imgs/style-1.png --input ./imgs/non-makeup.png --color_only`
- Pattern Only: `python main.py --style ./imgs/style-1.png --input ./imgs/non-makeup.png --pattern_only`

Result image will be saved in `result.png` ()


1. Pattern Makeup Transfer

---

### About Data

---

### Train & Test

Redirection to [Color Branch](./Color/README.md) and [Pattern Branch](./Pattern/README.md)


---

##### Common Issues

1. [Solved] `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`:


```
sudo apt update
sudo apt install libgl1-mesa-glx

```

1. [Solved] `RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 1 does not equal 0 (while checking arguments for cudnn_convolution)`

Add CUDA VISIBLE DEVICES `CUDA_VISIBLE_DEVICES=0 python main.py`

##### Checklists

**To-do**

- [ ] Usage
	- [x] Color
	- [x] Pattern
	- [x] C+P
	- [ ] Partial
- [ ] Train/ Test
	- [ ] Color
	- [ ] Pattern
- [ ] Data
	- [ ] ITW
	- [ ] Sticker
	- [ ] Create Synthesis Pipeline

**Issues**: ⚠️: important

- [ ] Check cuda() is available else cpu
- [ ] ⚠️ (Usage) blend_mode mask return noticable artifacts!!

---

##### Acknowledgements

Big thanks to [YadiraF (PRNet)](https://github.com/YadiraF/PRNet), [qubvel (segmentation_models.pytorch)](https://github.com/qubvel/segmentation_models.pytorch), and [wtjiang98 (BeautyGAN Pytorch)](https://github.com/wtjiang98/BeautyGAN_pytorch) for making theirs works publicly available.