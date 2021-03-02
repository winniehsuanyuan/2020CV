import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
img_num = len(objpoints)
p_per_img = corner_x*corner_y #49

world_coor = np.copy(objpoints)
camera_coor = np.zeros((img_num, p_per_img, 3))
#homo coor
for i in range(img_num):
    for j in range(p_per_img):
        world_coor[i][j][2] = 1
        camera_coor[i][j][0] = imgpoints[i][j][0][0]
        camera_coor[i][j][1] = imgpoints[i][j][0][1]
        camera_coor[i][j][2] = 1

#Pm=0 --> find H
H = []
for i in range(img_num):
    P = []
    for j in range(p_per_img):
        u, v = camera_coor[i][j][0], camera_coor[i][j][1]
        x, y = world_coor[i][j][0], world_coor[i][j][1]
        P.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        P.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    P = np.asarray(P)
    u, s, vh = np.linalg.svd(P)
    m = vh[-1,:] / vh[-1,-1]
    H.append(m.reshape(3, 3))

#Vb=0 --> find K
V=np.zeros((2*img_num ,6))
for i in range(img_num):
    V[2*i][0]=H[i][0][0]*H[i][0][1]
    V[2*i][1]=H[i][1][0]*H[i][0][1]+H[i][0][0]*H[i][1][1]
    V[2*i][2]=H[i][2][0]*H[i][0][1]+H[i][0][0]*H[i][2][1]
    V[2*i][3]=H[i][1][0]*H[i][1][1]
    V[2*i][4]=H[i][2][0]*H[i][1][1]+H[i][1][0]*H[i][2][1]
    V[2*i][5]=H[i][2][0]*H[i][2][1]
    V[2*i+1][0]=H[i][0][0]*H[i][0][0]-H[i][0][1]*H[i][0][1]
    V[2*i+1][1]=2*(H[i][0][0]*H[i][1][0]-H[i][0][1]*H[i][1][1])
    V[2*i+1][2]=2*(H[i][0][0]*H[i][2][0]-H[i][0][1]*H[i][2][1])
    V[2*i+1][3]=H[i][1][0]*H[i][1][0]-H[i][1][1]*H[i][1][1]
    V[2*i+1][4]=2*(H[i][1][0]*H[i][2][0]-H[i][1][1]*H[i][2][1])
    V[2*i+1][5]=H[i][2][0]*H[i][2][0]-H[i][2][1]*H[i][2][1]

u, s, vt = np.linalg.svd(V, full_matrices=True)
b=vt[-1,:] / vt[-1,-1]
B=np.zeros((3, 3))
B[0][0]=b[0]
B[0][1]=b[1]
B[0][2]=b[2]
B[1][1]=b[3]
B[1][2]=b[4]
B[2][2]=b[5]
B[1][0]=B[0][1]
B[2][0]=B[0][2]
B[2][1]=B[1][2]

#K
K=np.linalg.cholesky(B).T #actually it's K^(-1)

#extrinsic
extrinsics_S=np.zeros((img_num, 3, 4))
for x in range(img_num):
    l=1/(np.sum(np.power(np.matmul(K, H[x][:, 0]), 2))**0.5)
    extrinsics_S[x][:, 0]=l*np.matmul(K, H[x][:, 0])
    extrinsics_S[x][:, 1]=l*np.matmul(K, H[x][:, 1])
    extrinsics_S[x][:, 2]=np.cross(extrinsics_S[x][:, 0], extrinsics_S[x][:, 1])
    extrinsics_S[x][:, 3]=l*np.matmul(K, H[x][:, 2])

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
#camera_matrix=mtx
camera_matrix = np.linalg.inv(K)
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics_S, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()





