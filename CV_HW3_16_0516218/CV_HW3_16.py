#!/usr/bin/env python
# coding: utf-8

# In[23]:


import cv2
import numpy as np
import math

def dist(des1, des2):
    return math.sqrt(np.sum((des1-des2)**2))

img_1 = cv2.imread('data/1.jpg')
img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2 = cv2.imread('data/2.jpg')
img2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

#SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#feature matching: for every keypoint in image1, find 2 best matches
match_pt=np.zeros((des1.shape[0], 2)) #(indices of kp1, indices of kp2_1, indices of kp2_2)
match_dist=np.zeros((des1.shape[0], 2)) #(distance of kp1 and kp2_1, distance of kp1 and kp2_2)
for i in range(des1.shape[0]):
    min1=1000
    min2=1000
    for j in range(des2.shape[0]):
        d=dist(des1[i], des2[j])
        if d <= min1:
            min2=min1
            min1=d
            pt1=j
        elif d <= min2:
            min2=d
            pt2=j
    match_pt[i][0]=pt1
    match_pt[i][1]=pt2
    match_dist[i][0]=min1
    match_dist[i][1]=min2

#ratio distance: for the 2 best matches of each keypoint, if the ratio distance>0.45, discard
goodMatch=[]
coor1=[]
coor2=[]
int_coor1=[]
int_coor2=[]
for x in range(match_dist.shape[0]):
    if match_dist[x][0] < 0.45*match_dist[x][1]:
        goodMatch.append([x,int(match_pt[x][0])])

gM_num=len(goodMatch)
for m, n in goodMatch:
    coor1.append((kp1[m].pt[0],kp1[m].pt[1]))  
    coor2.append(((kp2[n].pt[0]+img1.shape[1]),kp2[n].pt[1]))
    int_coor1.append((int(kp1[m].pt[0]),int(kp1[m].pt[1]))) 
    int_coor2.append((int(kp2[n].pt[0]+img1.shape[1]),int(kp2[n].pt[1])))

#feature matching visualization
radius = 2
color_arr = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
thickness = 1
vis_img = np.concatenate((img_1, img_2), axis=1)   
for x in range(gM_num):
    color=color_arr[x%6]
    vis_img=cv2.circle(vis_img, int_coor1[x], radius, color, thickness)
    vis_img=cv2.circle(vis_img, int_coor2[x], radius, color, thickness)
    vis_img=cv2.line(vis_img, int_coor1[x], int_coor2[x], color, thickness)
cv2.imwrite('feature_matching_visualization.jpg', vis_img)

#find homography matrix H by RANSAC
#homo coor
homo_coor1=np.zeros((3, gM_num))
homo_coor2=np.zeros((3, gM_num))
for x in range(gM_num):
    homo_coor1[0,x] = coor1[x][0]
    homo_coor1[1,x] = coor1[x][1]
    homo_coor1[2,x] = 1
    homo_coor2[0,x] = coor2[x][0]-img1.shape[1]
    homo_coor2[1,x] = coor2[x][1]
    homo_coor2[2,x] = 1
#RANSAC
max_inliers=0
best_inliers=[]
for i in range(10): #iteration 
    P = []
    sample=np.random.randint(gM_num, size=4) #samples
    for j in sample:
        x, y = homo_coor1[0,j], homo_coor1[1,j]
        u, v = homo_coor2[0,j], homo_coor2[1,j]
        P.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        P.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    P = np.asarray(P)
    u, s, vh = np.linalg.svd(P)
    h = (vh[-1,:] / vh[-1,-1]).reshape(3, 3)
    cal_coor2=np.matmul(h, homo_coor1)
    cal_coor2[0,:]=cal_coor2[0,:]/cal_coor2[2,:]
    cal_coor2[1,:]=cal_coor2[1,:]/cal_coor2[2,:]
    cal_coor2[2,:]=cal_coor2[2,:]/cal_coor2[2,:]
    n=0
    inliers=[]
    for k in range(gM_num):
        if dist(homo_coor2[:,k], cal_coor2[:,k])<100: #distance threshold
            n=n+1
            inliers.append(k)
    if n>max_inliers:
        best_inliers=inliers
        max_inliers=n
#recalculate homography matrix
B=[] 
for b in best_inliers:
    x, y = homo_coor1[0,b], homo_coor1[1,b]
    u, v = homo_coor2[0,b], homo_coor2[1,b]
    B.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
    B.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
B = np.asarray(B)
u, s, vh = np.linalg.svd(B)
H = (vh[-1,:] / vh[-1,-1]).reshape(3, 3) 

#warp image
wrap_img=np.zeros((img_1.shape[0],img_1.shape[1]+img_2.shape[1], 3))
for i in range(wrap_img.shape[0]):
    for j in range(wrap_img.shape[1]):
        c=np.matmul(H, np.array([j, i, 1]).T)
        c=(c/c[2])
        c=np.round(c)
        if np.all(c>=0) and c[0]<img_2.shape[1] and c[1]<img_2.shape[0]:
            wrap_img[i, j, :]=img_2[int(c[1]),int(c[0]), :]
wrap_img[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
cv2.imwrite('wrap_img.jpg',wrap_img)

