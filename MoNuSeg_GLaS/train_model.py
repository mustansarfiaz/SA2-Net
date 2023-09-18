# -*- coding: utf-8 -*-
# @Time    : 2022/12/19 8:59 
# @Author  : Mustansar Fiaz
# @File    : train.py

import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from sklearn.model_selection import KFold
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D
#from nets.UCTransNet import UCTransNet
from nets.sasanet import MyNet
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model=='val_best':
        filename = save_path + '/' + \
                   'val_best_model-{}.pth.tar'.format(model)
    elif best_model=='test_best':
        filename = save_path + '/' + \
                   'test_best_model-{}.pth.tar'.format(model)
        
    else:
        filename = save_path + '/' + \
                   'model-{}-last.pth.tar'.format(model)
    torch.save(state, filename)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          Main Loop: load model,
#=================================================================================
##################################################################################
def main_loop(train_loader,val_loader, test_loader, batch_size=config.batch_size, model_type='', fold=0, tensorboard=True, kfold=0):
#def main_loop(batch_size=config.batch_size, model_type='', fold=0, tensorboard=True, kfold=0):
    # Load train and val data
    

    lr = config.learning_rate
    logger.info(model_type)


    config_vit = config.get_CTranS_config()
    logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
    logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
    logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
    model = MyNet(n_channels=config.n_channels,n_classes=config.n_labels)
    
    model = model.cuda()
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])
    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler =  None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    max_dice_test = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= {} | Fold [{}/{}] | Epoch [{}/{}] ========='.format(config.model_name,fold,kfold, epoch + 1, config.epochs + 1))        
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        #train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, fold, kfold, logger)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            #val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
            #                                optimizer, writer, epoch, lr_scheduler,model_type,logger)
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                         optimizer, writer, epoch, lr_scheduler,fold,kfold,logger)
            test_loss, test_dice = train_one_epoch(test_loader, model, criterion,
                                                         optimizer, writer, epoch, lr_scheduler,fold,kfold,logger)


        # =============================================================
        #       Save best model
        # =============================================================
        if test_dice > max_dice_test:
            if epoch+1 > 2:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice_test = test_dice
                best_epoch = epoch + 1
                # save_checkpoint({'epoch': epoch,
                #                  'best_model': True,
                #                  'model': model_type,
                #                  'state_dict': model.state_dict(),
                #                  'val_loss': val_loss,
                #                  'optimizer': optimizer.state_dict()}, config.model_path)
                save_checkpoint({'epoch': epoch,
                                 'best_model': 'test_best',
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path+"fold_"+str(fold)+"/")
                
        
        if val_dice > max_dice:
            if epoch+1 > 5:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                # save_checkpoint({'epoch': epoch,
                #                  'best_model': True,
                #                  'model': model_type,
                #                  'state_dict': model.state_dict(),
                #                  'val_loss': val_loss,
                #                  'optimizer': optimizer.state_dict()}, config.model_path)
                save_checkpoint({'epoch': epoch,
                                 'best_model': 'val_best',
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path+"fold_"+str(fold)+"/")
                
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        save_checkpoint({'epoch': epoch,
                         'best_model': 'Last',
                         'model': model_type,
                         'state_dict': model.state_dict(),
                         'val_loss': val_loss,
                         'optimizer': optimizer.state_dict()}, config.model_path+"fold_"+str(fold)+"/")
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return max_dice


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    
    
    if config.task_name == "Synapse":
        filelists = os.listdir(config.train_dataset)
    else:
        filelists = os.listdir(config.train_dataset+"img")

    filelists = np.array(filelists)
    kfold = config.kfold
    kf = KFold(n_splits=kfold, shuffle=True, random_state=config.seed)
    dice_list = []
    iou_list = []
    
    
    filelists_test = os.listdir(config.test_dataset+"/img")
    filelists_test = np.array(filelists_test)
    
    for fold, (train_index, val_index) in enumerate(kf.split(filelists)):
        train_filelists = filelists[train_index]
        val_filelists = filelists[val_index]
        np.savetxt(config.save_path+"val_fold_"+str(fold+1)+".txt", val_filelists,'%s')
        logger.info("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

        train_tf= RandomGenerator(output_size=[config.img_size, config.img_size2])

        val_tf = ValGenerator(output_size=[config.img_size, config.img_size2])
        train_dataset = ImageToImage2D(config.train_dataset,
                                             train_tf,
                                             image_size=config.img_size,
                                             filelists=train_filelists,
                                             task_name=config.task_name)
        val_dataset = ImageToImage2D(config.train_dataset,
                                            val_tf,
                                            image_size=config.img_size,
                                            filelists=val_filelists,
                                            task_name=config.task_name)
        test_dataset = ImageToImage2D(config.test_dataset,
                                            val_tf,
                                            image_size=config.img_size,
                                            filelists=filelists_test,
                                            task_name=config.task_name)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn,
                                  num_workers=0,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                worker_init_fn=worker_init_fn,
                                num_workers=0,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                worker_init_fn=worker_init_fn,
                                num_workers=0,
                                pin_memory=True)
        
        dice = main_loop(train_loader,val_loader, test_loader,  model_type=config.model_name, fold=fold+1, tensorboard=True, kfold=kfold)
        dice_list.append(dice.item())

    dice=0.0
    for j in range(len(dice_list)):
        logging.info("fold {0}: {1:2.4f}".format(j+1, dice_list[j]))
        dice+=dice_list[j]
    logging.info("mean dice: {:.4f} \n".format(dice/kfold))



