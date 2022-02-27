import time
import math
import sys

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from imageio import imread, imwrite
from glob import glob
from io import BytesIO
import requests
import matplotlib.pyplot as plt

import st_helper
import utils
import PIL

def np_to_pil(npy):
    return PIL.Image.fromarray(npy.astype(np.uint8))

def pil_to_np(pil):
    return np.array(pil)

def show_img(img):
    # Code for displaying at actual resolution from:
    # https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
    dpi = 80
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def run_st(content_path, style_path, content_weight, max_scl, coords, use_guidance,regions, output_path='output.png'):

    smll_sz = 64
    
    start = time.time()

    content_im_big = utils.to_device(Variable(utils.load_path_for_pytorch(content_path,512,force_scale=True).unsqueeze(0)))

    for scl in range(1,max_scl):

        long_side = smll_sz*(2**(scl-1))
        lr = 2e-3

        ### Load Style and Content Image ###
        content_im = utils.to_device(Variable(utils.load_path_for_pytorch(content_path,long_side,force_scale=True).unsqueeze(0)))
        content_im_mean = utils.to_device(Variable(utils.load_path_for_pytorch(style_path,long_side,force_scale=True).unsqueeze(0))).mean(2,keepdim=True).mean(3,keepdim=True)
        
        ### Compute bottom level of laplaccian pyramid for content image at current scale ###
        lap = content_im.clone()-F.upsample(F.upsample(content_im,(content_im.size(2)//2,content_im.size(3)//2),mode='bilinear'),(content_im.size(2),content_im.size(3)),mode='bilinear')
        nz = torch.normal(lap*0.,0.1)

        canvas = F.upsample( lap, (content_im_big.size(2),content_im_big.size(3)), mode='bilinear')[0].data.cpu().numpy().transpose(1,2,0)

        if scl == 1:
            canvas = F.upsample(content_im,(content_im.size(2)//2,content_im.size(3)//2),mode='bilinear')[0].data.cpu().numpy().transpose(1,2,0)

        ### Initialize by zeroing out all but highest and lowest levels of Laplaccian Pyramid ###
        if scl == 1:
            if 1:
                stylized_im = Variable(content_im_mean+lap)
            else:
                stylized_im = Variable(content_im.data)

        ### Otherwise bilinearly upsample previous scales output and add back bottom level of Laplaccian pyramid for current scale of content image ###
        if scl > 1 and scl < max_scl-1:
            stylized_im = F.upsample(stylized_im.clone(),(content_im.size(2),content_im.size(3)),mode='bilinear')+lap

        if scl == max_scl-1:
            stylized_im = F.upsample(stylized_im.clone(),(content_im.size(2),content_im.size(3)),mode='bilinear')
            lr = 1e-3

        ### Style Transfer at this scale ###
        stylized_im, final_loss = st_helper.style_transfer(stylized_im, content_im, style_path, output_path, scl, long_side, 0., use_guidance=use_guidance, coords=coords, content_weight=content_weight, lr=lr, regions=regions)

        canvas = F.upsample(stylized_im,(content_im.size(2),content_im.size(3)),mode='bilinear')[0].data.cpu().numpy().transpose(1,2,0)
        
        ### Decrease Content Weight for next scale ###
        content_weight = content_weight/2.0

    print("Finished in: ", int(time.time()-start), 'Seconds')
    print('Final Loss:', final_loss)

    canvas = torch.clamp( stylized_im[0], 0., 1.).data.cpu().numpy().transpose(1,2,0)
    result_image = tensor_to_np(tensor_resample(canvas))

    # renormalize image
    result_image -= result_image.min()
    result_image /= result_image.max()

    result = np_to_pil(result_image * 255.)
    show_img(pil_to_np(result))
    imwrite(output_path,canvas)
    return final_loss , canvas

if __name__=='__main__':

    ### Parse Command Line Arguments ###
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    content_weight = float(sys.argv[3])*16.0
    max_scl = 5

    use_guidance_region = '-gr' in sys.argv
    use_guidance_points = False
    use_gpu = not ('-cpu' in sys.argv)
    utils.use_gpu = use_gpu


    paths = glob(style_path+'*')
    losses = []
    ims = []


    ### Preprocess User Guidance if Required ###
    coords=0.
    if use_guidance_region:
        i = sys.argv.index('-gr')
        regions = utils.extract_regions(sys.argv[i+1],sys.argv[i+2])
    else:
        try:
            regions = [[imread(content_path)[:,:,0]*0.+1.], [imread(style_path)[:,:,0]*0.+1.]]
        except:
            regions = [[imread(content_path)[:,:]*0.+1.], [imread(style_path)[:,:]*0.+1.]]

    ### Style Transfer and save output ###
    loss,canvas = run_st(content_path,style_path,content_weight,max_scl,coords,use_guidance_points,regions,output_path=sys.argv[5])
