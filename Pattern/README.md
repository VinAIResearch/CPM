# Pattern Makeup - Training Guideline

| ![pattern_segmentation.png](../imgs/pattern_segmentation.png) | 
|:--:| 
| Pattern Makeup | Pattern Segmentation Model|

This is training guideline for Pattern Branch (P), one out of two branches in [Color-Pattern Makeup Transfer (CPM)](../README.md).

---

1. **Requirements**: Please refer to [Getting Started/ Requirements](../README.md), the main components are:
	- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
	- torch >=1.6
1. **Data Preparation**: Please download [CPM-Synt-1](../readme-about-data.md)
1. **Training**: `python train.py --datapath /pathtodata`
1. (Optional):
	- Open Tensorboard: `tensorboard --logdir=runs`
	- Change backbones, pretrained_weights: Check `parser.py`

### Acknowledgements

This code is based on [PRNet](https://github.com/YadiraF/PRNet) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)