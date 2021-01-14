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
	- Color Makeup Branch (CM)
	- Pattern Makeup Branch (CP)

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

##### Usage

- Color Makeup Transfer: `python main.py --color`

- Result image will be saved in `result.png`

1. Pattern Makeup Transfer

---

### About Data

---

### Train & Test

Redirection to [Color Branch]() and [Pattern Branch]()


---

##### Common Issues

1. [Solved] `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`:


```
sudo apt update
sudo apt install libgl1-mesa-glx

```

1. [Solved] `RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 1 does not equal 0 (while checking arguments for cudnn_convolution)`

Change to `CUDA_VISIBLE_DEVICES=0 python main.py --pattern_only --style ./imgs/style-1.png`

##### Checklists

1. To-do
- [ ] Usage
	- [x] Color
	- [ ] Pattern
	- [ ] C+P
	- [ ] Partial
- [ ] Train/ Test
	- [ ] Color
	- [ ] Pattern
- [ ] Data
	- [ ] ITW
	- [ ] Sticker
	- [ ] Create Synthesis Pipeline
1. Issues
- [ ] Check cuda() is available else cpu

---

##### Acknowledgements

Big thanks to [YadiraF (PRNet)](https://github.com/YadiraF/PRNet), [qubvel (segmentation_models.pytorch)](https://github.com/qubvel/segmentation_models.pytorch) and [wtjiang98 (BeautyGAN Pytorch)](https://github.com/wtjiang98/BeautyGAN_pytorch) for making theirs works publicly available.