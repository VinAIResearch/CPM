# Pattern Makeup - Training Guideline

| ![color-makeup.png](../imgs/color-makeup.png) | 
|:--:| 
| This is training guideline for Color Branch (P), one out of two branches in [Color-Pattern Makeup Transfer (CPM)](../README.md).|

---

For Color Branch, we used the same CycleGAN-based model like [BeautyGAN](liusi-group.com/pdf/BeautyGAN-camera-ready_2.pdf).
But instead of normal training pair, we used our **novel uv-space**.

1. Create dataset: Download [Makeup Transfer Dataset](http://liusi-group.com/projects/BeautyGAN). Use [PRNet](https://github.com/YadiraF/PRNet) to get respective uv-map texture of each image and its segmentation mask. More detail should be find in [Synthesis-Process](../Synthesis-Process/README.md)
1. Train: Follow instruction at [BeautyGAN-pytorch](https://github.com/wtjiang98/BeautyGAN_pytorch). Re-train model with newly established dataset.