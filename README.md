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

Result image will be saved in `result.png` (style | original image | result)

---

### About Data

We introduce ✨ 4 new datasets: CPM-Real, CPM-Synt-1, CPM-Synt-2, and Stickers datasets. Besides, we also use published [LADN's Dataset](https://georgegu1997.github.io/LADN-project-page/) & [Makeup Transfer Dataset](http://liusi-group.com/projects/BeautyGAN).

|    Name  						  | #imgs | Description						   | - 									|
|:-------------------------------:|:-----:|:-----------------------------------|:----------------------------------:|
|[CPM-Real](./readme-about-data.md)| 3895  | real - makeup styles 			   |![CPM-Real.png](./imgs/CPM-Real.png)|
|[CPM-Synt-1](./readme-about-data.md)| 5555| synthesis - makeup img with pattern segmentation mask|![./imgs/CPM-Synt-1.png](./imgs/CPM-Synt-1.png)|
|[CPM-Synt-2](./readme-about-data.md)| 1625| synthesis - triplets: makeup, non-makeup, ground-truth|![./imgs/CPM-Synt-2.png](./imgs/CPM-Synt-2.png)|
|[Stickers](./readme-about-data.md)|577| high-quality transparent imgs, used to create CPM-Synt-1/2 |![Stickers.png](./imgs/Stickers.png)|

Please refer to [readme-about-data.md](./readme-about-data.md) for downloading these datasets.

---

### Train & Test


As stated in the paper, the Color Branch and Pattern Branch are totally independent. Yet, they shared the same workflow:

1. Data preparation: Use [PRNet](https://github.com/YadiraF/PRNet) to generate texture_map of faces.
1. Training

Please redirect to [Color Branch 🇨](./Color) or [Pattern Branch 🇵](./Pattern) for further details

---

### Common Issues

1. [Solved] `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`:


```
sudo apt update
sudo apt install libgl1-mesa-glx
```

1. [Solved] `RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 1 does not equal 0 (while checking arguments for cudnn_convolution)`

Add CUDA VISIBLE DEVICES `CUDA_VISIBLE_DEVICES=0 python main.py`

### Update Log 

##### Checklist

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

##### Issues

⚠️: important

- [ ] [usage] check cuda() is available else cpu
- [ ] ⚠️ [usage] blend_mode mask return noticable artifacts!!

---

### Acknowledgements

Big thanks to [YadiraF (PRNet)](https://github.com/YadiraF/PRNet), [qubvel (segmentation_models.pytorch)](https://github.com/qubvel/segmentation_models.pytorch), and [wtjiang98 (BeautyGAN Pytorch)](https://github.com/wtjiang98/BeautyGAN_pytorch) for making theirs works publicly available.