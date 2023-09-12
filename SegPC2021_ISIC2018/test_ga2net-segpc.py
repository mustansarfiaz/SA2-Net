#!/usr/bin/env python
# coding: utf-8

# # GA2Net - SegPC2021
# ---

# ## Import packages & functions

# In[1]:


from __future__ import print_function, division


import os
import sys
sys.path.append('../..')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import copy
import json
import importlib
import glob
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim import Adam, SGD
from losses import DiceLoss, DiceLossWithLogtis
from torch.nn import BCELoss, CrossEntropyLoss

from utils import (
    show_sbs,
    load_config,
    _print,
)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode


# ## Set the seed

# In[2]:


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
import random
random.seed(0)


# ## Load the config

# In[5]:


CONFIG_NAME = "segpc/segpc2021_ga2net.yaml"
CONFIG_FILE_PATH = os.path.join("./configs", CONFIG_NAME)


# In[6]:


config = load_config(CONFIG_FILE_PATH)
_print("Config:", "info_underline")
print(json.dumps(config, indent=2))
print(20*"~-", "\n")


# ## Dataset and Dataloader

# In[8]:


from datasets.segpc import SegPC2021Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


# In[9]:


# ----------------- transform ------------------
# transform for image
img_transform = transforms.Compose([
    transforms.ToTensor()
])
# transform for mask
msk_transform = transforms.Compose([
    transforms.ToTensor()
])
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ----------------- dataset --------------------
# preparing training dataset
tr_dataset = SegPC2021Dataset(
    mode="tr",
    input_size=config['dataset']['input_size'],
    scale=config['dataset']['scale'],
    data_dir=config['dataset']['data_dir'], 
    dataset_dir=config['dataset']['dataset_dir'],
    one_hot=True,
    force_rebuild=False,
    img_transform=img_transform,
    msk_transform=msk_transform
)
# preparing training dataset
vl_dataset = SegPC2021Dataset(
    mode="vl",
    input_size=config['dataset']['input_size'],
    scale=config['dataset']['scale'],
    data_dir=config['dataset']['data_dir'], 
    dataset_dir=config['dataset']['dataset_dir'],
    one_hot=True,
    force_rebuild=False,
    img_transform=img_transform,
    msk_transform=msk_transform
)
# preparing training dataset
te_dataset = SegPC2021Dataset(
    mode="te",
    input_size=config['dataset']['input_size'],
    scale=config['dataset']['scale'],
    data_dir=config['dataset']['data_dir'], 
    dataset_dir=config['dataset']['dataset_dir'],
    one_hot=True,
    force_rebuild=False,
    img_transform=img_transform,
    msk_transform=msk_transform
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

# prepare train dataloader
tr_dataloader = DataLoader(tr_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['train'])

# prepare validation dataloader
vl_dataloader = DataLoader(vl_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['validation'])

# prepare test dataloader
te_dataloader = DataLoader(te_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['test'])



print(f"Length of trainig_dataset:\t{len(tr_dataset)}")
print(f"Length of validation_dataset:\t{len(vl_dataset)}")
print(f"Length of test_dataset:\t\t{len(te_dataset)}")

# -------------- test -----------------
# test and visualize the input data
for batch in tr_dataloader:
    img = batch['image']
    msk = batch['mask']
    print("Training")
    show_sbs(img[0,:-1,:,:], msk[0,1])
    break
    
for batch in vl_dataloader:
    img = batch['image']
    msk = batch['mask']
    print("Validation")
    show_sbs(img[0,:-1,:,:], msk[0,1])
    break
    
for batch in te_dataloader:
    img = batch['image']
    msk = batch['mask']
    print("Test")
    show_sbs(img[0,:-1,:,:], msk[0,1])
    break


# ### Device

# In[10]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Torch device: {device}")


# ## Metrics

# In[11]:


metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.F1Score(num_classes=config['dataset']['number_classes'], task='multiclass'),
        torchmetrics.Accuracy(num_classes=config['dataset']['number_classes'], task='multiclass'),
        torchmetrics.Dice(),
        torchmetrics.Precision(num_classes=config['dataset']['number_classes'], task='multiclass'),
        torchmetrics.Specificity(num_classes=config['dataset']['number_classes'], task='multiclass'),
        torchmetrics.Recall(num_classes=config['dataset']['number_classes'], task='multiclass'),
        # IoU
        torchmetrics.JaccardIndex(num_classes=config['dataset']['number_classes'], task='multiclass'),
    ],
    prefix='train_metrics/'
)

# train_metrics
train_metrics = metrics.clone(prefix='train_metrics/').to(device)

# valid_metrics
valid_metrics = metrics.clone(prefix='valid_metrics/').to(device)

# test_metrics
test_metrics = metrics.clone(prefix='test_metrics/').to(device)


# In[12]:


def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res



# ## Define test function

# In[18]:


def test(model, te_dataloader):
    model.eval()
    with torch.no_grad():
        evaluator = test_metrics.clone().to(device)
        for batch_data in tqdm(te_dataloader):
            imgs = batch_data['image']
            msks = batch_data['mask']
            
            imgs = imgs.to(device)
            msks = msks.to(device)
            
            preds = model(imgs)
            
            # we should remove nucs
            not_nucs = torch.where(imgs[:,-1,:,:]>0, 0, 1)
            preds_en = torch.argmax(preds, 1, keepdim=False).float() * not_nucs
            msks_en = torch.argmax(msks, 1, keepdim=False) * not_nucs
            
            # evaluate by metrics
            evaluator.update(preds_en, msks_en.int())
    return evaluator


# ## Load and prepare model

# In[21]:


from models.ga2net.gaganet import MyNet as Net
import models._uctransnet.Config as uct_config
config_vit = uct_config.get_CTranS_config()

# # Test the best inferred model
# ----

# ## Load the best model

# In[25]:


best_model = Net(n_channels=config['model']['params']['n_channels'],
            n_classes=config['model']['params']['n_classes'])


torch.cuda.empty_cache()
best_model = best_model.to(device)

fn = "best_model_state_dict.pt"
os.makedirs(config['model']['save_dir'], exist_ok=True)
model_path = f"{config['model']['save_dir']}/{fn}"

best_model.load_state_dict(torch.load(model_path))
print("Loaded best model weights...")


# ## Evaluation

# In[26]:


te_metrics = test(best_model, te_dataloader)
print('**********************    Scores are ****************')
print('*****************************************************')
print('*****************************************************')
print(te_metrics.compute())
print('*****************************************************')
print('*****************************************************')


# ## Plot graphs

# In[27]:


result_file_path = f"{config['model']['save_dir']}/result.json"
#result_file_path = './saved_models/segpc2021_uctransnet/result.json'
with open(result_file_path, 'r') as f:
    results = json.loads(''.join(f.readlines()))
epochs_info = results['epochs_info']

tr_losses = [d['tr_loss'] for d in epochs_info]
vl_losses = [d['vl_loss'] for d in epochs_info]
tr_dice = [d['tr_metrics']['train_metrics/Dice'] for d in epochs_info]
vl_dice = [d['vl_metrics']['valid_metrics/Dice'] for d in epochs_info]
tr_js = [d['tr_metrics']['train_metrics/MulticlassJaccardIndex'] for d in epochs_info]
vl_js = [d['vl_metrics']['valid_metrics/MulticlassJaccardIndex'] for d in epochs_info]
tr_acc = [d['tr_metrics']['train_metrics/MulticlassAccuracy'] for d in epochs_info]
vl_acc = [d['vl_metrics']['valid_metrics/MulticlassAccuracy'] for d in epochs_info]


_, axs = plt.subplots(1, 4, figsize=[16,3])

axs[0].set_title("Loss")
axs[0].plot(tr_losses, 'r-', label="train loss")
axs[0].plot(vl_losses, 'b-', label="validatiton loss")
axs[0].legend()

axs[1].set_title("Dice score")
axs[1].plot(tr_dice, 'r-', label="train dice")
axs[1].plot(vl_dice, 'b-', label="validation dice")
axs[1].legend()

axs[2].set_title("Jaccard Similarity")
axs[2].plot(tr_js, 'r-', label="train JaccardIndex")
axs[2].plot(vl_js, 'b-', label="validatiton JaccardIndex")
axs[2].legend()

axs[3].set_title("Accuracy")
axs[3].plot(tr_acc, 'r-', label="train Accuracy")
axs[3].plot(vl_acc, 'b-', label="validation Accuracy")
axs[3].legend()

plt.show()


# In[28]:


epochs_info


# ## Save images

# In[31]:


from PIL import Image


save_imgs_dir = f"{config['model']['save_dir']}/visualized"

if not os.path.isdir(save_imgs_dir):
    os.mkdir(save_imgs_dir)

with torch.no_grad():
    for batch in tqdm(te_dataloader):
        imgs = batch['image']
        msks = batch['mask']
        ids = batch['id']
        
        preds = best_model(imgs.to(device))
        
        txm = imgs.cpu().numpy()
        tbm = torch.argmax(msks, 1).cpu().numpy()
        tpm = torch.argmax(preds, 1).cpu().numpy()
        tid = ids.cpu().numpy()
        
        for idx in range(len(tbm)):
            img = np.moveaxis(txm[idx, :3], 0, -1)*255.
            nuc = np.uint8(txm[idx, -1]*255.)
            gt = np.uint8(tbm[idx]*255.)
            gt_3ch = np.stack([gt-nuc, nuc*0, nuc], -1)
            pred = np.where(tpm[idx]>0.5, 255, 0)
            pred_3ch = np.stack([pred-nuc, nuc*0, nuc], -1)
            
            img = np.ascontiguousarray(img, dtype=np.uint8)
            gt_3ch = np.ascontiguousarray(gt_3ch, dtype=np.uint8)
            pred_3ch = np.ascontiguousarray(pred_3ch, dtype=np.uint8)
            
            fid = tid[idx]
            Image.fromarray(img).save(f"{save_imgs_dir}/{fid}_img.png")
            Image.fromarray(gt_3ch).save(f"{save_imgs_dir}/{fid}_gt.png")
            Image.fromarray(pred_3ch).save(f"{save_imgs_dir}/{fid}_pred.png")


# In[32]:


f"{config['model']['save_dir']}/visualized"

