import numpy as np
import glob
import os 
from  PIL import Image
import random
import cv2
from matplotlib import pyplot as plt


from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy import ndimage




def do_cyto_pred_process(pred):
    return pred

def get_cyto_mask(pred, th):
    mask = np.where(pred>th, 1, 0)
    return mask
    
    


def do_cyto_postprocess(mask, KS):
    t = np.uint8(np.where(mask, 255, 0))
    t = binary_erosion(t, structure=np.ones((KS,KS)))
    t = binary_dilation(t, structure=np.ones((KS,KS)))
    t = np.uint8(t)
    t = ndimage.binary_fill_holes(np.where(t>0,1,0), structure=np.ones((3,3))).astype(int)
    
    num_labels, labels_msk = cv2.connectedComponents(t.astype(np.uint8))
    for i in range(1, num_labels):
        idx, idy = np.where(labels_msk==i)
        labels_msk[labels_msk==i] = 0 if np.sum(np.where(labels_msk==i,1,0))<10 else i
    return np.where(labels_msk>0, 1, 0)



def get_biggest_cc_msk(mask):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[:, -1]
    
    if len(sizes)<2:
        return mask
    
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2




def get_pure_img_bbox(img):
    xmin, ymin, xmax, ymax = [0,0,img.shape[0]-1,img.shape[1]-1]
    
    while not np.sum(img[xmin, :]): xmin += 1
    while not np.sum(img[:, ymin]): ymin += 1
    while not np.sum(img[xmax, :]): xmax -= 1
    while not np.sum(img[:, ymax]): ymax -= 1
        
    bbox = [xmin, xmax+1, ymin, ymax+1]
    return bbox


def sim_resize(x, size, interpolation=cv2.INTER_NEAREST):
    if len(x.shape) == 3:
        if x.shape[-1] > 3:
            x2 = np.zeros_like(x)
            x2[:,:,:3] = cv2.resize(x[:,:,:3], size[::-1], interpolation=cv2.INTER_LINEAR )   
            x2[:,:, 3] = cv2.resize(x[:,:, 3], size[::-1], interpolation=cv2.INTER_NEAREST)
        else:
            x2 = cv2.resize(x, size[::-1], interpolation=cv2.INTER_LINEAR)
    else:
        x2 = cv2.resize(x, size[::-1], interpolation=cv2.INTER_NEAREST)
    return x2



# def resize(x, size=(512, 512)):
#     h, w = size
#     if x.ndim == 4:
#         x2 = np.zeros((x.shape[0], h, w, 3))
#     else:
#         x2 = np.zeros((x.shape[0], h, w))
#     for idx in range(len(x)):
#         x2[idx] = cv2.resize(x[idx], size, interpolation=cv2.INTER_NEAREST)   
#     return x2


# def sim_resize(x, size, interpolation=cv2.INTER_NEAREST):
#     x2 = cv2.resize(x, size[::-1], interpolation=interpolation)   
#     return x2
    
    
def resize(x, size, interpolation="nearest"):
    if interpolation.lower() == "linear":
        x2 = cv2.resize(x, size[::-1], interpolation=cv2.INTER_LINEAR) 
    else:
        x2 = cv2.resize(x, size[::-1], interpolation=cv2.INTER_NEAREST) 
    return x2

def resize_pad(img, size):
    sh = img.shape
    if sh[0]<size[0] and sh[1]<size[1]:
        if len(sh)==3:
            img_s = np.zeros((size[0], size[1], sh[-1]), dtype=np.uint8)
            shift_x = (img_s.shape[0] - sh[0])//2
            shift_y = (img_s.shape[1] - sh[1])//2
            img_s[shift_x:sh[0]+shift_x, shift_y:sh[1]+shift_y, :] = img
        else:
            img_s = np.zeros(size, dtype=np.uint8)
            shift_x = (img_s.shape[0] - sh[0])//2
            shift_y = (img_s.shape[1] - sh[1])//2
            img_s[shift_x:sh[0]+shift_x, shift_y:sh[1]+shift_y] = img
    else:
        img_s = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR) 
#         img_s = sim_resize(img, size, interpolation=cv2.INTER_NEAREST)
    return img_s


def show_sbs(iml, imr):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(iml, interpolation='none')
    plt.subplot(1,2,2)
    plt.imshow(imr, interpolation='none')
    return
    
    
def clean_nuc_mask(mask, KS=3):
    t = np.uint8(np.where(mask, 255, 0))
    t = binary_erosion(t, structure=np.ones((KS,KS)))
    t = binary_dilation(t, structure=np.ones((KS,KS)))
    t = np.uint8(t)
    t = ndimage.binary_fill_holes(np.where(t>0,1,0), structure=np.ones((3,3))).astype(int)
    
    num_labels, labels_msk = cv2.connectedComponents(t.astype(np.uint8))
    for i in range(1, num_labels):
        idx, idy = np.where(labels_msk==i)
        labels_msk[labels_msk==i] = 0 if np.sum(np.where(labels_msk==i,1,0))<80 else i
    return np.where(labels_msk>0, 1, 0)

    
    
def crop_multi_scale_submats(image, name, mask, desire_margin_list=[0,]):
    '''
    output: -> [dict]
        'meta': -> [string] => general information about image and mask and instances,
        'data': -> [list]   => # of individual nucleus
            <a list for nucleus 1: -> [list] => pre scale (margin)>
                {
                    'scale'------: -> [number] => scale,
                    'bbox'-------: -> [list]   => bbox,
                    'bbox_hint'--: -> [string] => "[x_min, y_min, x_max, y_max]",
                    'shift'------: -> [list]   => [shift_x, shift_y],
                    'simg_size'--: -> [list]   => snmsk.shape,
                    'simg'-------: -> [mat]    => simg,
                    'snmsk'------: -> [mat]    => snmsk,
                }
                :
                :
                {...}
            <a list for nucleus 2: -> [list] => pre scale (margin)>,
            :
            :
            <last nucleus .....>
    '''
    
    img = image
    msk = mask
    t = np.uint8(np.where(msk, 255, 0))
    num_labels, labels_msk = cv2.connectedComponents(t)

    all_inst_data_list = []
    for i in range(1, num_labels):
        msk = np.where(labels_msk==i,255,0)
        
        idxs, idys = np.where(labels_msk==i)
        n_i_bbox = [min(idxs), min(idys), max(idxs)+1, max(idys)+1]
        
        ## crop the nucleus
        bbox = [
            max(0           ,n_i_bbox[0]), max(0           ,n_i_bbox[1]),
            min(img.shape[0],n_i_bbox[2]), min(img.shape[1],n_i_bbox[3]),
        ]
        n_i_img = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        n_i_msk = msk[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
        all_scales_data_list = []
        for scale in desire_margin_list:
            dx = round(scale*n_i_msk.shape[0]/2)
            dy = round(scale*n_i_msk.shape[1]/2)
            
            snmsk = np.zeros((n_i_msk.shape[0]+2*dx,n_i_msk.shape[1]+2*dy)  ,dtype=np.uint8)
            simg  = np.zeros((n_i_msk.shape[0]+2*dx,n_i_msk.shape[1]+2*dy,4),dtype=np.uint8)
            
            bbox = [
                max(0           ,n_i_bbox[0]-dx),max(0           ,n_i_bbox[1]-dy),
                min(img.shape[0],n_i_bbox[2]+dx),min(img.shape[1],n_i_bbox[3]+dy),
            ]       
            
            timg  = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            tnmsk = msk[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            
            shift_x = round((simg.shape[0]-timg.shape[0])/2)
            shift_y = round((simg.shape[1]-timg.shape[1])/2)
            simg [shift_x:timg.shape[0] +shift_x, shift_y:timg.shape[1] +shift_y, :] = timg
            snmsk[shift_x:tnmsk.shape[0]+shift_x, shift_y:tnmsk.shape[1]+shift_y]    = tnmsk
            
            simg[:,:,3] = simg[:,:,3] * np.where(snmsk>0,1,0)
                    
            tdata = {
                'scale': scale,
                'bbox': bbox,
                'bbox_hint': "[x_min, y_min, x_max, y_max]",
                'shift': [shift_x, shift_y],
                'simg_size': snmsk.shape,
                'simg': simg,
                'snmsk': snmsk,
            }            
            all_scales_data_list.append(tdata)

        all_inst_data_list.append(all_scales_data_list)
    
    data = {
        'meta': {
            'image_size' : img.shape,
            'image_name' : name,
            'total_insts': len(all_inst_data_list)
        },
        'data': all_inst_data_list
    }

    return data

















