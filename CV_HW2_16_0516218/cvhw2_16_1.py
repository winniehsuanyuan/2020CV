#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import cv2
import numpy as np
import math

def imgResize(img1,img2):
    diff_x=abs(img1.shape[0]-img2.shape[0])
    diff_y=abs(img1.shape[1]-img2.shape[1])
    
    if img1.shape[0]>img2.shape[0]:
        lower=int(diff_x/2)
        upper=img1.shape[0]-lower
        if (diff_x%2)==1:
            upper=upper-1
        img1=img1[lower:upper, :,:]
    else:
        lower=int(diff_x/2)
        upper=img2.shape[0]-lower
        if (diff_x%2)==1:
            upper=upper-1
        img2=img2[lower:upper, :,:]
            
    if img1.shape[1]>img2.shape[1]:
        lower=int(diff_y/2)
        upper=img1.shape[1]-lower
        if (diff_y%2)==1:
            upper=upper-1
        img1=img1[:,lower:upper,:]
    else: 
        lower=int(diff_y/2)
        upper=img2.shape[1]-lower
        if (diff_y%2)==1:
            upper=upper-1
        img2=img2[:,lower:upper,:]
    return img1, img2


def shift(img):
    shift_img=np.zeros((img.shape))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for z in range(img.shape[2]):
                shift_img[x,y,z]=img[x,y,z]*((-1)**(x+y))
    return shift_img
    
def dist(x, y):
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def idealFilter(D0, img, lowPass):
    filtered=np.zeros((img.shape), dtype = complex)
    center=(img.shape[0]/2, img.shape[1]/2)
    if lowPass:
        f=np.zeros((img.shape[:2]))
    else:
        f=np.ones((img.shape[:2]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if dist(center, (x,y))<D0:
                f[x,y]=int(lowPass)
    for z in range(img.shape[2]):
        filtered[:,:,z]=img[:,:,z]*f
    return filtered

def gaussianFilter(D0, img, lowPass):
    filtered=np.zeros((img.shape), dtype = complex)
    center=(img.shape[0]/2, img.shape[1]/2)
    f=np.zeros((img.shape[:2]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            f[x,y]=math.exp(((-(dist((x,y),center))**2)/(2*(D0**2))))
    if lowPass==False:
        f=1-f
    for z in range(img.shape[2]):
        filtered[:,:,z]=img[:,:,z]*f
    return filtered

def ft(img):
    f_img=np.zeros((img.shape), dtype = complex)
    for z in range(img.shape[2]):
        f_img[:,:,z]=np.fft.fft2(img[:,:,z])
    return f_img

def ift(img):
    if_img=np.zeros((img.shape), dtype = complex)
    for z in range(img.shape[2]):
        if_img[:,:,z]=np.fft.ifft2(img[:,:,z])
    return if_img  

###########################################################    
# Define different cutoff frequencies for each pair of pics
L_D0=[15,10,10,15,10,10,10,15,20]
H_D0=[30,20,25,15,25,25,30,20,25]

for i in range(0,7):
    images = glob.glob('hw2_data/task1and2_hybrid_pyramid/'+str(i)+'*')
    img1=cv2.imread(images[1])#
    img2=cv2.imread(images[0])#

    # Resize if different image sizes
    if np.any(img1.shape!=img2.shape):
        img1, img2=imgResize(img1, img2)

    # Multiply each channel of the input image by (-1)^(x+y)
    # Do 2D Fourier transformation on each channel
    f_img1=ft(shift(img1))
    f_img2=ft(shift(img2))
   
    # Ideal filter
    iLP_fimg=idealFilter(L_D0[i], f_img1, lowPass=True)       
    iHP_fimg=idealFilter(H_D0[i], f_img2, lowPass=False)      #highPass

    # Gaussian filter
    gLP_fimg=gaussianFilter(L_D0[i], f_img1, lowPass=True)   
    gHP_fimg=gaussianFilter(H_D0[i], f_img2, lowPass=False)   #highPass

    # Do 2D inverse Fourier transformation on each channel
    # Obtain the real parts
    # Multiply each channel of the filtered image by (-1)^(x+y)
    iLP_img=shift(ift(iLP_fimg).real)
    iHP_img=shift(ift(iHP_fimg).real)
    gLP_img=shift(ift(gLP_fimg).real)
    gHP_img=shift(ift(gHP_fimg).real)

    # Hybrid
    ihybrid_img=iLP_img+iHP_img
    ghybrid_img=gLP_img+gHP_img

    # Save
    cv2.imwrite(str(i)+'_i_x.jpg', ihybrid_img)
    cv2.imwrite(str(i)+'_g_x.jpg', ghybrid_img)

