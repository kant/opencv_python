import cv
import sys
import cv2
import numpy as np
#import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
 


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


#inputImg1 = cv2.imread('00000062_edge.jpg')
#inputImg2 = cv2.imread('00000063_edge.jpg')

inputImg1 = cv2.imread('00000062.jpg')
inputImg2 = cv2.imread('00000063.jpg')

       
input_img_prev = cv2.cvtColor(inputImg1, cv2.COLOR_BGR2GRAY)
input_img_curr = cv2.cvtColor(inputImg2, cv2.COLOR_BGR2GRAY)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(input_img_prev)
input_img_prev = cl1;

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl2 = clahe.apply(input_img_curr)
input_img_curr = cl2;

cv2.namedWindow('input_img_prev', cv2.WINDOW_AUTOSIZE)
cv2.imshow('input_img_prev',input_img_prev)


cv2.namedWindow('input_img_curr', cv2.WINDOW_AUTOSIZE)
cv2.imshow('input_img_curr',input_img_curr)

print ('input_img_prev size', input_img_prev.shape)
#cv2.SetImageROI(input_img_prev, [bbox_x1,bbox_x2,bbox_y1,bbox_y2])
#input_img_prev = input_img_prev[bbox_x1:bbox_x1+height,bbox_y1:bbox_y1+width]
#input_img_curr = input_img_curr[bbox_x1:bbox_x1+height,bbox_y1:bbox_y1+width]
#input_img_prev = input_img_prev[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
#input_img_curr = input_img_curr[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

# --------------------  SURF Points --------------- #
surf = cv2.SURF(25)
e1 = cv2.getTickCount()
kp, des = surf.detectAndCompute(input_img_prev,None)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print ('kp', len(kp))
input_img_prev_surf = cv2.drawKeypoints(input_img_prev,kp,None,(255,0,0),4)

cv2.namedWindow('SURF Points', cv2.WINDOW_AUTOSIZE)
img_prev_RGB = inputImg1.copy()
str_temp = 'SURF Points in ROI: '
str_temp += str(len(kp))
str_temp += ' | Time: '
str_temp += str(time)[:5]
str_temp += ' s'
cv2.putText(input_img_prev_surf,str_temp,(10,30), font, 0.5,(255,255,255),2)
cv2.imshow('SURF Points',input_img_prev_surf) 
pt2 = []
pt2_narray = np.empty((0,2), float)
#print ('pt2_array', pt2_narray)

# Generate SURF points array
index = 0
for each_point in kp:
    pt2.append(each_point.pt)
    x = each_point.pt[0]
    y = each_point.pt[1]
    pt2_narray = np.vstack((pt2_narray, np.array([x, y])))
    index = index + 1
#   y.append(kp[i].y)
print index
pt2_narray =  np.float32(pt2_narray).reshape(-1, 1, 2)
print ('pt2_narray',type(pt2_narray))
print ('pt2_size', pt2_narray.shape)
# print ('pt2_narray', pt2_narray)

#x,y = kp[:].pt
#print ('x, y', x, y)
# ---------------------------------------------------- #
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                     qualityLevel = 0.3,
                     minDistance = 7,
                     blockSize = 7 )

#p0 = cv2.goodFeaturesToTrack(input_img_prev, mask = None, **feature_params)
#print ('p0_type', type(p0))
#print ('p0_size', p0.shape)
# print ('p0', p0)

p0 = pt2_narray;
p1, status, err = cv2.calcOpticalFlowPyrLK(input_img_prev, input_img_curr, p0, None, **lk_params)
print ('p1_type', type(p1))
print ('p1_size', p1.shape)
# print ('p1', p1)

p0r, status, err = cv2.calcOpticalFlowPyrLK(input_img_curr, input_img_prev, p1, None, **lk_params)
print ('p0r_type', type(p1))
print ('p0r_size', p1.shape)
# print ('p0r', p1)

d = abs(p0-p0r).reshape(-1, 2).max(-1)
print d

good = d < 1
#print ('d', d)

print ('good', good)
print ("After Forward-backward", good.shape)

good_curr_points = p1[status == 1]
good_prev_points = p0[status == 1]

# Create a mask image for drawing purposes
#mask = np.zeros_like(inputImg1)
#print ('mask size', mask.shape)

# Create some random colors
color = np.random.randint(0,255,(1000,3))
print ('color size', color.shape)

#img_prev_ROI_RGB = inputImg1[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
img_prev_ROI_RGB = inputImg1.copy()
img_prev_ROI_RGB_all_points = inputImg1.copy()

#img_curr_ROI_RGB = inputImg2[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
img_curr_ROI_RGB = inputImg2.copy()
img_curr_ROI_RGB_all_points = inputImg2.copy()

vis3_good = img_curr_ROI_RGB.copy()
# vis4_all = inputImg2.copy()
vis4_all = cv2.addWeighted(img_prev_ROI_RGB,0.5,img_curr_ROI_RGB,0.5,1)
#print ('p1', p1)

for i,(new, old, good_flag) in enumerate(zip(good_curr_points, good_prev_points, good)):
    
    a,b = new.ravel()
    cv2.circle(img_curr_ROI_RGB_all_points, (a, b), 3, color[i].tolist(), 1)

    c,d = old.ravel() 
    cv2.circle(img_prev_ROI_RGB_all_points, (c, d), 3, color[i].tolist(), 1)

    cv2.line(vis4_all, (a,b),(c,d), color[i].tolist(), 2)
    if not (good_flag):
        continue    
    cv2.line(vis3_good, (a,b),(c,d), color[i].tolist(), 2)

# Copy img_prev_ROI_RGB_all_points (ROI with Interested Points) to cl1
#Original_Img1 = cl1.copy()
#img_prev_RGB_all_points = inputImg1.copy()
#img_prev_RGB_all_points[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = img_prev_ROI_RGB_all_points
cv2.namedWindow('Previous Image All Points', cv2.WINDOW_AUTOSIZE)
str_temp = 'Prev. Img. | Good Points: '
str_temp += str(len(good_prev_points))
cv2.putText(img_prev_ROI_RGB_all_points,str_temp,(10,30), font, 0.5,(255,255,255),2)
cv2.imshow('Previous Image All Points',img_prev_ROI_RGB_all_points) 

# Copy img_prev_ROI_RGB_all_points (ROI with Interested Points) to cl2
#img_curr_RGB_all_points = inputImg2.copy()
#img_curr_RGB_all_points[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = img_curr_ROI_RGB_all_points
cv2.namedWindow('Current Image All Points', cv2.WINDOW_AUTOSIZE)
str_temp = 'Prev. Img. | Good Points: '
str_temp += str(len(good_prev_points))
cv2.putText(img_curr_ROI_RGB_all_points,str_temp,(10,30), font, 0.5,(255,255,255),2)
cv2.imshow('Current Image All Points',img_curr_ROI_RGB_all_points)

inputImg2_good_points_RGB = inputImg2.copy()
#inputImg2_good_points_RGB[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = vis3_good
cv2.namedWindow('Matched Good Points', cv2.WINDOW_AUTOSIZE)
str_temp = 'Matched Good Points: '
str_temp += str(sum(good))
cv2.putText(vis3_good,str_temp,(10,30), font, 0.5,(255,255,255),2)
cv2.imshow('Matched Good Points',vis3_good) 

inputImg2_all_points_RGB = inputImg2.copy()
#inputImg2_all_points_RGB[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = vis4_all
cv2.namedWindow('All Matched Points', cv2.WINDOW_AUTOSIZE)
str_temp = 'Matched All Points: '
str_temp += str(len(good_curr_points))
cv2.putText(vis4_all,str_temp,(10,30), font, 0.5,(255,255,255),2)
cv2.imshow('All Matched Points',vis4_all)

# ------------------------------ Dense Flow -----------------------------------------
e1 = cv2.getTickCount()
flow = cv2.calcOpticalFlowFarneback(input_img_prev,input_img_curr,0.5,1,3,15,3,5,1)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
 # print ('flow', flow)
print ('flow_size', flow.shape)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])


#img_prev_ROI_RGB = inputImg1[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
img_prev_ROI_RGB = inputImg1.copy()
dens_points_gray = np.zeros_like(img_prev_ROI_RGB)
num_largest = 200
indices = (-mag).argpartition(num_largest, axis=None)[:num_largest]
x, y = np.unravel_index(indices, mag.shape)
print ('x=', x)
print ('y=', y)
for i in range(0, num_largest):
  cv2.circle(dens_points_gray, (y[i], x[i]), 3, color[i].tolist(), -1)
#img1_original_rgb = np.zeros_like(inputImg1)
#img1_original_rgb[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = dens_points_gray
cv2.namedWindow('Optical Flow Points', cv2.WINDOW_AUTOSIZE)
str_temp = 'ROI Dense Optical Flow Selected '
str_temp += str(num_largest)
str_temp += ' Points'
cv2.putText(dens_points_gray,str_temp,(10,30), font, 0.5,(255,255,255),2)
cv2.imshow('Optical Flow Points',dens_points_gray)

# Draw HSV Flow Code
#img_prev_ROI_RGB = inputImg1[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
img_prev_ROI_RGB = inputImg1.copy()
hsv = np.zeros_like(img_prev_ROI_RGB)
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

print ('hsv_size', hsv.shape)
print ('hsv_type', type(hsv))
print ('inputImg1', inputImg1.shape)
print ('inputImg1_type', type(inputImg1))
 
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#img1_original_rgb = np.zeros_like(inputImg1)
#img1_original_rgb[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = rgb
cv2.namedWindow('Optical Flow HSV', cv2.WINDOW_AUTOSIZE)
str_temp = 'ROI Dense Optical Flow'
str_temp += ' | Time: '
str_temp += str(time)[:5]
str_temp += ' s'
cv2.putText(rgb,str_temp,(10,30), font, 0.5,(255,255,255),2)
cv2.imshow('Optical Flow HSV',rgb)

# cv2.imshow('flow', draw_flow(input_img_prev, flow))


cv2.waitKey(0)
cv2.destroyAllWindows()
