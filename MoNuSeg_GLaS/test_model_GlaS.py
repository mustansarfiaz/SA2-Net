import torch.optim
from Load_Dataset_GlaS import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config_GlaS as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
#from nets.UCTransNet import UCTransNet
from nets.gaganet import MyNet
from utils import *
import cv2
import numpy as np

import time

def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name is "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(2000,2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred

def show_ens(predict_save,input_img, labs, save_path):
    fig, ax = plt.subplots()
    plt.imshow(predict_save, cmap='gray')
    plt.axis("off")
    height, width = predict_save.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=300)
    plt.close()

def vis_and_save_heatmap(ensemble_models, input_img, img_RGB, labs, lab_img, vis_save_path):
    
    outputs = []
    dice_pred, iou_pred = [],[]
    for model in ensemble_models:
        output = model(input_img.cuda())
        pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
        predict_save = pred_class[0].cpu().data.numpy()
        outputs.append(predict_save)
        predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
        dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
        dice_pred.append(dice_pred_tmp)
        iou_pred.append(iou_tmp)
        
    predict_save = np.array(outputs).mean(0)
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    predict_save = np.where(predict_save>0.5,1,0)
    show_ens(predict_save, img_RGB, lab_img, save_path=vis_save_path+'_pred5f_'+model_type+'.jpg')
    return dice_pred, iou_pred
  


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ensemble_models=[]
    test_session = config.test_session
    for i in range(0,5):
        if config.task_name is "GLaS":
            test_num = 80
            model_type = config.model_name
            #model_path = "./GlaS/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
            model_path = "./GLaS/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/val_best_model-"+model_type+".pth.tar"
    
        elif config.task_name is "MoNuSeg":
            test_num = 14
            model_type = config.model_name
            #model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
            model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/val_best_model-"+model_type+".pth.tar"
    
    
        save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
        vis_path = "./" + config.task_name + '_visualize_test/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    
        maxi = 5
        if not os.path.exists(model_path):
            maxi = i
            print("====",maxi, "models loaded ====")
            break
        checkpoint = torch.load(model_path, map_location='cuda')
    
    
        config_vit = config.get_CTranS_config()
        model = MyNet(n_channels=config.n_channels,n_classes=config.n_labels)
    
        model = model.cuda()
        # if torch.cuda.device_count() > 1:
        #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        #     model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded !')
        model.eval()
        ensemble_models.append(model)
    filelists = os.listdir(config.test_dataset+"/img")
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    #test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_dataset = ImageToImage2D(config.test_dataset,
                                    tf_test,
                                    image_size=config.img_size,
                                    task_name=config.task_name,
                                    filelists=filelists,
                                    split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    dice_pred = np.zeros((maxi))
    iou_pred = np.zeros((maxi))
    dice_class = np.zeros((maxi,8))
    dice_5folds = []
    iou_5folds = []
    end = time.time()
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()
            
            img_RGB = cv2.imread(config.test_dataset+"img/"+names[0],1)
            img_RGB = cv2.resize(img_RGB,(config.img_size,config.img_size))


            lab_img = cv2.imread(config.test_dataset+"labelcol/"+names[0][:-4]+"_anno.bmp",0)
            lab_img = cv2.resize(lab_img,(config.img_size,config.img_size))
            
            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t = vis_and_save_heatmap(ensemble_models, input_img,  img_RGB, lab,lab_img,
                                                          vis_path+str(i))
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    inference_time = (time.time() - end)/test_num
    print("inference_time",inference_time)
    dice_pred = dice_pred/test_num * 100.0
    iou_pred = iou_pred/test_num * 100.0
    if config.n_labels > 1:
        dice_class = dice_class/test_num * 100.0
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    print ("dice_5folds:",dice_pred)
    print ("iou_5folds:",iou_pred)
    dice_pred_mean = dice_pred.mean()
    iou_pred_mean = iou_pred.mean()
    if config.n_labels > 1:
        dice_class_mean = dice_class.mean(0)
    dice_pred_std = np.std(dice_pred,ddof=1)
    iou_pred_std = np.std(iou_pred,ddof=1)
    print ("dice: {:.2f}+{:.2f}".format(dice_pred_mean, dice_pred_std))
    print ("iou: {:.2f}+{:.2f}".format(iou_pred_mean, iou_pred_std))
    if config.n_labels > 1:
        np.set_printoptions(formatter={'float': '{:.2f}'.format})
        print ("dice class:",dice_class_mean)






