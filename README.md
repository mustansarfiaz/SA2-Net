# SA2-Net: Scale-aware Attention Network for Microscopic Image Segmentation (BMVC'23 -- Oral)

Highlights
-----------------
- ** Microscopic image segmentation network:** The development of SA2-Net, a microscopic image segmentation network comprising an encoder and a decoder.
change detection (CD).
- **scale-aware attention (SA2) module:** The incorporation of the scale-aware attention (SA2) module within the decoder. This module effectively captures scale and shape variations by applying local scale attention at each stage and global scale attention across scales using multi-resolution features.
- **Adaptive up-attention (AuA) module:** The implementation of progressive refinement and upsampling of scale-enhanced features through the adaptive up-attention (AuA) module, leading to the generation of the final segmentation mask.

Methods
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/ScratchFormer/blob/main/demo/proposed_framework.jpg">

Introduction
-----------------
In this study, we introduce SA2-Net, a novel attention-guided approach that harnesses multi-scale feature learning to effectively handle the diverse structures found within microscopic images. Our key innovation is the scale-aware attention (SA2) module, specially designed to capture inherent variations in the scales and shapes of microscopic regions, such as cells, enabling precise segmentation. This SA2 module incorporates local attention mechanisms at each level of the multi-stage features and global attention mechanisms that span across multiple resolutions.

Additionally, we tackle the challenge of blurry region boundaries, such as cell boundaries, by introducing a groundbreaking upsampling strategy called the Adaptive Up-Attention (AuA) module. This innovative module enhances discriminative capabilities, leading to improved localization of microscopic regions through the application of an explicit attention mechanism.

## Contact
If you have any questions, please create an issue on this repository or contact us at mustansar.fiaz@mbzuai.ac.ae

<hr />

## References
Our code is based on [Awesome-U-Net](https://github.com/NITR098/Awesome-U-Net),  [UCTransNet](https://github.com/McGregorWwww/UCTransNet), and [CASCADE](https://github.com/SLDGroup/CASCADE/tree/main)   repositories. We thank them for releasing their baseline code.

For MoNuSeg and GlaS, we follow [UCTransNet](https://github.com/McGregorWwww/UCTransNet).
For ISIC2018 and SegPC2021, we follow [Awesome-U-Net](https://github.com/NITR098/Awesome-U-Net).
For ACDC, we follow [CASCADE](https://github.com/SLDGroup/CASCADE/tree/main).




## Citation

```
@inproceedings{fiaz2022sat,
  title={SA2-Net: Scale-aware Attention Network for Medical Image Segmentation},
  author={Fiaz, Mustansar and Heidari, Moein and Anwar, Rao Muhammad and Cholakkal, Hisham},
  booktitle={BMVC},
  year={2023}
}
