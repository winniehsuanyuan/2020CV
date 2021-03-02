import numpy as np
import sys
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
import time
from scipy.signal import convolve2d as c2d
import os
import time
path = 'hw2_data\\task3_colorizing\\'
def Get_three_channel(img):
   
    img = sk.img_as_float(img)
    
   
    h = int(img.shape[0] / 3)

    # separate color channels
    blue = img[:h]
    green = img[h: 2 * h]
    red = img[2 * h: 3 * h]
    return [blue, green, red]

def Crop_Img(image, m=0.1):
    
    h, w = image.shape
    y1, y2 = int(m * h), int((1 - m) * h)
    x1, x2 = int(m * w), int((1 - m) * w)
    return image[y1:y2, x1:x2]

def Sum_of_Squared_Defferences(a, b):
    
    output = np.sum((a- b) ** 2)
    return output

def align(img, b, k_size):
    
    m = np.zeros((k_size * 2, k_size * 2))
    for i in range(-k_size, k_size):
        for j in range(-k_size, k_size):
            new_img = np.roll(img, i, axis=0)
            new_img = np.roll(new_img, j, axis=1)
            ssd_val = Sum_of_Squared_Defferences(b, new_img)
            m[i + k_size, j + k_size] = ssd_val
    if m != []:
        min_m = m.argmin()
        row_shift = (min_m // (k_size * 2)) - k_size
        col_shift = (min_m % (k_size * 2)) - k_size
    else:
        row_shift = 0
        col_shift = 0
  
    return (row_shift, col_shift)

def alignGandRtoB(r, g, b, k_size):
   
    g_align_b = align(g, b, k_size)
    r_align_b = align(r, b, k_size)
    return (g_align_b, r_align_b)

def Colorizing_the_Russian_Empire_tif(file):
    
    img = skio.imread(file)
    height = img.shape[0] // 3
    Three_Channel = Get_three_channel(img)
    for i in range(len(Three_Channel)):
        Three_Channel[i] = Crop_Img(Three_Channel[i])

   
    reconstruct_b = Three_Channel[0]
    reconstruct_g = Three_Channel[1]
    reconstruct_r = Three_Channel[2]

 
 
    down_scale = 20
    k_size = int((height // down_scale) // 5)
    g_shift_row, g_shift_col, r_shift_row, r_shift_col = 0, 0, 0, 0

    while (down_scale >= 1):
       
        smaller_b = sktr.downscale_local_mean(Three_Channel[0], (down_scale, down_scale))
        smaller_g = sktr.downscale_local_mean(Three_Channel[1], (down_scale, down_scale))
        smaller_r = sktr.downscale_local_mean(Three_Channel[2], (down_scale, down_scale))
        

        
        Shift_result = alignGandRtoB(smaller_r, smaller_g, smaller_b,k_size)
        Shift_g = Shift_result[0]
        Shift_r = Shift_result[1]        

        
        g_shift_row += (Shift_g[0] * down_scale)
        g_shift_col += (Shift_g[1] * down_scale)
        r_shift_row += (Shift_r[0] * down_scale)
        r_shift_col += (Shift_r[1] * down_scale)

        Three_Channel[1] = np.roll(Three_Channel[1], Shift_g[0] * down_scale, axis=0)
        Three_Channel[1] = np.roll(Three_Channel[1], Shift_g[1] * down_scale, axis=1)
        Three_Channel[2] = np.roll(Three_Channel[2], Shift_r[0] * down_scale, axis=0)
        Three_Channel[2] = np.roll(Three_Channel[2], Shift_r[1] * down_scale, axis=1)

        down_scale = down_scale // 2
        k_size = k_size // 2
 
 

    reconstruct_g = np.roll(reconstruct_g, g_shift_row, axis=0)
    reconstruct_g = np.roll(reconstruct_g, g_shift_col, axis=1)
    reconstruct_r = np.roll(reconstruct_r, r_shift_row, axis=0)
    reconstruct_r = np.roll(reconstruct_r, r_shift_col, axis=1)

    output = np.dstack([reconstruct_r, reconstruct_g, reconstruct_b])

    
  
    fname = 'aftre'+file[-8:-4]+'.jpg'
    skio.imsave(fname, output)
    skio.imshow(output)
    skio.show()

def Colorizing_the_Russian_Empire_jpg(file):
    
    img = skio.imread(file)
    height = img.shape[0] // 3
    Three_Channel = Get_three_channel(img)
    for i in range(len(Three_Channel)):
        Three_Channel[i] = Crop_Img(Three_Channel[i])
   
    reconstruct_b = Three_Channel[0]
    reconstruct_g = Three_Channel[1]
    reconstruct_r = Three_Channel[2]
 
    k_size = 15
    g_shift_row, g_shift_col, r_shift_row, r_shift_col = 0, 0, 0, 0


    Shift_result = alignGandRtoB(reconstruct_r, reconstruct_g, reconstruct_b,k_size)
    Shift_g = Shift_result[0]
    Shift_r = Shift_result[1]        
    #print(Shift_result)
    
    g_shift_row = Shift_g[0]
    g_shift_col = Shift_g[1]
    r_shift_row = Shift_r[0]
    r_shift_col = Shift_r[1]

    reconstruct_g = np.roll(reconstruct_g, g_shift_row, axis=0)
    reconstruct_g = np.roll(reconstruct_g, g_shift_col, axis=1)
    reconstruct_r = np.roll(reconstruct_r, r_shift_row, axis=0)
    reconstruct_r = np.roll(reconstruct_r, r_shift_col, axis=1)
    
    output = np.dstack([reconstruct_r, reconstruct_g, reconstruct_b])

    
    # save the alinged image
    fname = 'aftre'+file[-8:-4]+'.jpg'
    skio.imsave(fname, output)
    skio.imshow(output)
    skio.show()
for file in os.listdir(path):
    start_time = time.time()
    if file[-3:] == "tif" :
        
        Colorizing_the_Russian_Empire_jpg(path + "//" + file)
        print("--- %s seconds ---" % (time.time() - start_time))
        break
        
    if file[-3:] == "jpg" : 
        Colorizing_the_Russian_Empire_jpg(path + "//" + file)
        print(file)