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
1. [Train & Test](#train---test)
1. [Acknowledgements](#acknowledgements)
1. [Miscellaneous](#miscellaneous)

---

### Getting Started

##### Requirements

1. Clone the repo:
	```
	git clone https://github.com/VinAIResearch/CPM.git
	cd CPM
	```
1. Install packages `conda env create -f environment.yml`
1. Download Makeup pretrained models from [Drive](https://drive.google.com/drive/folders/1dagiuultGgDd_QNikMTrNlmCmWEaFV_N?usp=sharing). They are `pattern.pth` and `color.pth`. Put them in `checkpoints` folder.
1. Download [PRNet pretrained model] from [Drive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view). Put it in `PRNet/net-data`
##### Usage

- Color+Pattern: `python main.py --style ./imgs/style-1.png --input ./imgs/non-makeup.png`
- Color Only: `python main.py --style ./imgs/style-1.png --input ./imgs/non-makeup.png --color_only`
- Pattern Only: `python main.py --style ./imgs/style-1.png --input ./imgs/non-makeup.png --pattern_only`

Result image will be saved in `result.png` (style | original image | result)

---

### About Data

We introduce ✨ 4 new datasets: CPM-Real, CPM-Synt-1, CPM-Synt-2, and Stickers datasets. Besides, we also use published [LADN's Dataset](https://georgegu1997.github.io/LADN-project-page/) & [Makeup Transfer Dataset](http://liusi-group.com/projects/BeautyGAN).

Please refer to [readme-about-data.md](./readme-about-data.md) for downloading these datasets.

---

### Train & Test


As stated in the paper, the Color Branch and Pattern Branch are totally independent. Yet, they shared the same workflow:

1. Data preparation: Use [PRNet](https://github.com/YadiraF/PRNet) to generate texture_map of faces.
1. Training

Please redirect to [***Color Branch***](./Color) or [***Pattern Branch***](./Pattern) for further details.

---

### Acknowledgements

Big thanks to [YadiraF (PRNet)](https://github.com/YadiraF/PRNet), [qubvel (segmentation_models.pytorch)](https://github.com/qubvel/segmentation_models.pytorch), and [wtjiang98 (BeautyGAN Pytorch)](https://github.com/wtjiang98/BeautyGAN_pytorch) for making theirs works publicly available.

---

# Miscellaneous

### Checklist

- [x] Requirements | Packages
- [ ] Usage
	- [x] Color Only
	- [x] Pattern Only
	- [x] C+P
	- [ ] Partial | Mixed
- [ ] Train/ Test
	- [ ] Color (Train)
	- [ ] Pattern (Train | Test)
- [ ] Data | Pending (Added [readme-about-data.md](./readme-about-data.md))
	- [ ] CPM-Synt-1 | CPM-Synt-2
	- [ ] Stickers
	- [ ] CPM-Real
	- [ ] **Create Synthesis Pipeline**

### Issues 🛠️

🆘: important, need to fix ASAP | ⚠️: warning, minor bug | ✔️: fixed | 👍 currently fine

- **Usage**
	- ⚠️ check cuda() is available else cpu
	- 🆘 **blend_mode mask return noticable artifacts!!**
- **Training Code**
	- *Color*
	- *Pattern*
- **Data**
	- Synthesis Pipeline

### Trouble Shooting

1. [Solved] `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`:
	```
	sudo apt update
	sudo apt install libgl1-mesa-glx
	```
1. [Solved] `RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 1 does not equal 0 (while checking arguments for cudnn_convolution)`
	Add CUDA VISIBLE DEVICES before .py. Ex: `CUDA_VISIBLE_DEVICES=0 python main.py`

---