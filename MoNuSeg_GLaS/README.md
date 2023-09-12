# [BMVC2023] SA2Net


This repo is the official implementation of
['SA2-Net: Scale-aware Attention Network for Microscopic Image Segmentation']() which is accepted at BMVC2023 [Oral Presentation].

## Requirements

Install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```

### 1. Data Preparation
#### 1.1. GlaS and MoNuSeg Datasets
The original data can be downloaded in following links:
* MoNuSeg Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
* GLAS Dataset - [Link (Original)](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)

Then prepare the datasets in the following format for easy use of the code:
```angular2html
├── datasets
    ├── GlaS
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── MoNuSeg
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```

### 2. Training
The first step is to change the settings in ```Config.py```,
all the configurations including learning rate, batch size and etc. are 
in it.
Run:
```angular2html
python train_model.py
```

### 3. Testing
First, change the session name in ```Config.py``` as the training phase.
Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 


## Reference


* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)



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
