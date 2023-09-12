# SA2-Net: Scale-aware Attention Network for Microscopic Image Segmentation (BMVC'23 -- Oral)


This repo is the official implementation of
['SA2-Net: Scale-aware Attention Network for Microscopic Image Segmentation']() which is accepted at BMVC2023 [Oral Presentation].

### The structure of codes

Here is

```bash
.
├── README.md
├── images
│   └── *.png
├── configs
│   ├── isic
│   │   ├── isic2018_*<net_name>.yaml
│   └── segpc
│       └── segpc2021_*<net_name>.yaml
├── datasets
│   ├── *<dataset_name>.py
│   ├── *<dataset_name>.ipynb
│   └── prepare_*<dataset_name>.ipynb
├── models
│   ├── *<net_name>.py
│   └── _*<net_name>
│       └── *.py
├── train_and_test
│   ├── isic
│   │   ├── *<net_name>-isic.ipynb
│   │   └── *<net_name>-isic.py
│   └── segpc
│       ├── *<net_name>-segpc.ipynb
│       └── *<net_name>-segpc.py
├── losses.py
└── utils.py
```



## Dataset prepration

Please go to ["./datasets/README.md"](https://github.com/NITR098/Awesome-U-Net/blob/main/datasets/README.md) for details. We used 3 datasets for this work. After preparing required data you need to put the required data path in relevant config files.



## Train and Test

In the `train_and_test` folder, there are folders with the names of different datasets. In each of these subfolders, there are files related to each model network in two different formats (`.py` and ‍`.ipynb`). In notebook files you will face with the following procedures. This file contains both the testing and traning steps.

- Prepration step
  - Import packages & functions
  - Set the seed
  - Load the config file
- Dataset and Dataloader
  - Prepare Metrics
- Define test and validate function
- Load and prepare model
- Traning
  - Save the best model
- Test the best inferred model
  - Load the best model
- Evaluation
  - Plot graphs and print results
- Save images

## Citations


If this code is helpful for your study, please cite:
```
@inproceedings{fiaz2022sat,
  title={SA2-Net: Scale-aware Attention Network for Medical Image Segmentation},
  author={Fiaz, Mustansar and Heidari, Moein and Anwar, Rao Muhammad and Cholakkal, Hisham},
  booktitle={BMVC},
  year={2023}
}
```


## Contact 
If you have any issues, please raise an issue or contact at ([mustansar.fiaz@mbzuai.ac.ae](mustansar.fiaz@mbzuai.ac.ae))
