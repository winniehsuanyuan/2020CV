#!/usr/bin/env python
# coding: utf-8

import matlab.engine
import cv2
import numpy as np
import math

def dist(des1, des2):
    return math.sqrt(np.sum((des1-des2)**2))

img_1 = cv2.imread('Mesona1.JPG')
#img_1 = cv2.imread('Statue1.bmp')
img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2 = cv2.imread('Mesona2.JPG')
#img_2 = cv2.imread('Statue2.bmp')
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
    coor2.append((kp2[n].pt[0],kp2[n].pt[1]))
    int_coor1.append((int(kp1[m].pt[0]),int(kp1[m].pt[1]))) 
    int_coor2.append((int(kp2[n].pt[0]),int(kp2[n].pt[1])))

#estimate the fundamental matrix with 8-point algorithm
#homo coor
homo_coor1=np.zeros((3, gM_num))
homo_coor2=np.zeros((3, gM_num))
for x in range(gM_num):
    homo_coor1[0,x] = coor1[x][0]
    homo_coor1[1,x] = coor1[x][1]
    homo_coor1[2,x] = 1
    homo_coor2[0,x] = coor2[x][0]
    homo_coor2[1,x] = coor2[x][1]
    homo_coor2[2,x] = 1
#normalization
T1=np.array([[2/img1.shape[0], 0, -1],[0, 2/img1.shape[1], -1],[0, 0, 1]])
T2=np.array([[2/img2.shape[0], 0, -1],[0, 2/img2.shape[1], -1],[0, 0, 1]])
norm_coor1=T1@homo_coor1
norm_coor2=T2@homo_coor2

#RANSAC
max_inliers=0
best_inliers=[]
for i in range(100): #iteration 
    A = []
    sample=np.random.randint(gM_num, size=8) #samples
    for j in sample:
        u, v = norm_coor1[0,j], norm_coor1[1,j]
        x, y = norm_coor2[0,j], norm_coor2[1,j]
        A.append([u*x, u*y, u, v*x, v*y, v, x, y, 1])
    A = np.asarray(A)
    u, s, vh = np.linalg.svd(A)
    f = (vh[-1,:] / vh[-1,-1]).reshape(3, 3)
    #constraint det(F)=0
    u, s, vh = np.linalg.svd(f)
    s[2]=0
    F=u@np.diag(s)@vh
    #|x'TFx|<threshold to determine inliers
    n=0
    inliers=[]
    for k in range(gM_num):
        if abs((norm_coor2[:, k].T@F)@norm_coor1[:,k])<1: #threshold
            n=n+1
            inliers.append(k)
    if n>max_inliers:
        best_inliers=inliers
        max_inliers=n
        
#recalculate fundamental matrix
B=[] 
for b in best_inliers:
    u, v = norm_coor1[0,b], norm_coor1[1,b]
    x, y = norm_coor2[0,b], norm_coor2[1,b]
    B.append([u*x, u*y, u, v*x, v*y, v, x, y, 1])
B = np.asarray(B)
u, s, vh = np.linalg.svd(B)
f = (vh[-1,:] / vh[-1,-1]).reshape(3, 3) 
u, s, vh = np.linalg.svd(f)
s[2]=0
F=u@np.diag(s)@vh
#denormalize
F=T2.T@F@T1

#features & epipolar lines visualization
radius = 3
color_arr = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
thickness = 1
pt_img=img_1.copy()
eline_img=img_2.copy()
for x in range(gM_num):
    color=color_arr[x%6]
    pt_img=cv2.circle(pt_img, int_coor1[x], radius, color, thickness)
    m=F@homo_coor1[:,x]
    y1=int((-m[0]*0-m[2])/m[1])
    y2=int((-m[0]*(img2.shape[1]-1)-m[2])/m[1])
    eline_img=cv2.line(eline_img, (0, y1),(img2.shape[1]-1, y2), color, thickness)
vis_img = np.concatenate((pt_img, eline_img), axis=1)
cv2.imwrite('feature_epipolar_visualization.jpg', vis_img)

#Essential matrix
K = np.array([[1.4219,0.0005,0.5092],[0,1.4219,0.3802],[0,0,0.01]])
E = K.T @ F @ K
'''
K1 = np.array([[5426.566895,0.678017,330.096680],
               [0.000000,5423.133301,648.950012],
               [0.000000,0.000000,1.000000]])
K2 = np.array([[5426.566895,0.678017,387.430023],
               [0.000000,5423.133301,620.616699],
               [0.000000,0.000000,1.000000]])
E = K2.T @ F @ K1
'''

#Camera matrix
u, s, vh = np.linalg.svd(E)
m=1
s=np.array([m, m, 0])
E=u @ np.diag(s) @ vh
u, s, vh = np.linalg.svd(E)
W=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

R1=np.identity(3)
c1=np.zeros(3).reshape(3,1)
R21=u @ W @ vh
R22=u @ W.T @ vh
c21=np.array(u[:, 2])
c21 = c21.reshape(3,1)
c22=np.array(-u[:, 2])
c22 = c22.reshape(3,1)

P_o = np.concatenate((K @ R1, K @ R1 @ -c1),axis=1) 
P1 = np.concatenate((K @ R21, K @ R21 @ -c21),axis=1) 
P2 = np.concatenate((K @ R21, K @ R21 @ -c22),axis=1) 
P3 = np.concatenate((K @ R22, K @ R22 @ -c21),axis=1) 
P4 = np.concatenate((K @ R22, K @ R22 @ -c22),axis=1) 
'''
P_o = np.concatenate((K1 @ R1, K1 @ R1 @ -c1),axis=1) 
P1 = np.concatenate((K2 @ R21, K2 @ R21 @ -c21),axis=1) 
P2 = np.concatenate((K2 @ R21, K2 @ R21 @ -c22),axis=1) 
P3 = np.concatenate((K2 @ R22, K2 @ R22 @ -c21),axis=1) 
P4 = np.concatenate((K2 @ R22, K2 @ R22 @ -c22),axis=1) 
'''

#Triangulation
def triangulation(c1,c2,P1,R1,T1,P2,R2,T2):
    X_3d = []
    X_2d = []
    n= 0
    for i in range(gM_num):
        A = np.array([c1[i][0] * P1[2,:] - P1[0,:],
                      c1[i][1] * P1[2,:] - P1[1,:],
                      c2[i][0] * P2[2,:] - P2[0,:],
                      c2[i][1] * P2[2,:] - P2[1,:]
                     ])
        Au, As, Avh = np.linalg.svd(A)
        #3d point
        p=Avh[-1,:3]/Avh[-1,-1]
        X_3d.append(p.tolist())
        X_2d.append(c1[i])
        #camera center
        C1=-R1.T@T1
        C2=-R2.T@T2
        #check if the 3d point is in front of both cameras
        if ((np.dot((p - C1.T),R1[2,:].T)) > 0) and ((np.dot((p - C2.T),R2[2,:].T)) > 0):
            n += 1
    return n,X_3d,X_2d

#choose the solution with most 3d points in front of both cameras
max_n = (-1,[])
n1 = triangulation(coor1,coor2,P_o,R1,c1, P1,R21,c21)
if n1[0] > max_n[0]:
    max_n = n1
n2 = triangulation(coor1,coor2,P_o,R1,c1, P2,R21,c22)
if n2[0] > max_n[0]:
    max_n = n2
n3 = triangulation(coor1,coor2,P_o,R1,c1, P3,R22,c21)
if n3[0] > max_n[0]:
    max_n = n3
n4 = triangulation(coor1,coor2,P_o,R1,c1, P4,R22,c22)
if n4[0] > max_n[0]:
    max_n = n4

#3d reconstruction & texture mapping
eng = matlab.engine.start_matlab()
m = P_o.tolist()
m = matlab.double(m)
threedp =  matlab.double(max_n[1])
twodp =  matlab.double(max_n[2])
eng.obj_main(threedp, twodp, m, 'Mesona1.JPG', 1 ,nargout=0)
#eng.obj_main(threedp, twodp, m, 'Statue1.bmp', 1 ,nargout=0)

