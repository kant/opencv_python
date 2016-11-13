import cv
import sys
import cv2
import numpy as np
import csv
import math
#import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX

# initialize the list of reference points and booleani ndicating
# whether cropping is being performed or not
refPt = []
cropping = False

color = np.random.randint(0,255,(1000,3))
print ('color size', color.shape)


def click_and_crop(event, x, y, flags, param):
  # grab references to the global variables
  global refPt, cropping
 
  # if the left mouse button was clicked, record the starting
  # (x, y) coordinates and indicate that cropping is being
  # performed
  if event == cv2.EVENT_LBUTTONDOWN:
    refPt = [(x, y)]
    cropping = True
 
  #elif event == cv2.EVENT_MOUSEMOVE:
  #  if cropping == True:
  #    cv2.rectangle(clone,refPt[0],(x,y),(0,255,0,),2)

  # check to see if the left mouse button was released
  elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
    refPt.append((x, y))
    cropping = False
 
    # draw a rectangle around the region of interest
    cv2.rectangle(clone, refPt[0], refPt[1], (0, 255, 0), 2)
    cv2.imshow('input_img_prev', clone)


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

def squeeze_pts(X):
  X = X.squeeze()
  if len(X.shape) == 1:
    X = np.array([X])
  return X

def image_process(img_last_rgb, img_curr_rgb):
  global bb_x1, bb_y1, bb_x2, bb_y2
  global prev_bbx_x1, prev_bbx_y1, prev_bbx_x2, prev_bbx_y2
  global inputImg1, inputImg2, input_img_prev, input_img_curr
  global flag_predict_use_SURF, flag_predict_use_Dense

  flag_predict_use_SURF = False
  flag_predict_use_Dense = False
  flag_whole_imag_test = False
  # Get BBox Ground Truth by groudtruth
  #print index
  row = ground_truth_array[index]
  print ('bbox_gt:', row)

  if len(img_curr_rgb.shape) < 3:  
    inputImg1 = cv2.cvtColor(img_last_rgb, cv.CV_GRAY2RGB)
    inputImg2 = cv2.cvtColor(img_curr_rgb, cv.CV_GRAY2RGB)      
    input_img_prev = img_last_rgb.copy()
    input_img_curr = img_curr_rgb.copy()
  else:
    inputImg1 = img_last_rgb.copy()
    inputImg2 = img_curr_rgb.copy()
    input_img_prev = cv2.cvtColor(img_last_rgb, cv2.COLOR_BGR2GRAY)
    input_img_curr = cv2.cvtColor(img_curr_rgb, cv2.COLOR_BGR2GRAY)   


  if (flag_whole_imag_test == False):
    # Save All BBox file row to tmp variables
    tmp_x1 = int(row[0])
    tmp_y1 = int(row[1])
    tmp_x2 = int(row[2])
    tmp_y2 = int(row[3])
    tmp_x3 = int(row[4])
    tmp_y3 = int(row[5])
    tmp_x4 = int(row[6])
    tmp_y4 = int(row[7])
    print ('eight variables', tmp_x1, tmp_y1, tmp_x2, tmp_y2, tmp_x3, tmp_y3, tmp_x4, tmp_y4)
    # Selecet the top-left and bottom-right points, 
    # due to the different foramt(sequence) of the bbox file
    min_x = min(tmp_x1, tmp_x2, tmp_x3, tmp_x4)
    min_y = min(tmp_y1, tmp_y2, tmp_y3, tmp_y4)
    max_x = max(tmp_x1, tmp_x2, tmp_x3, tmp_x4)
    max_y = max(tmp_y1, tmp_y2, tmp_y3, tmp_y4)
    print ('minX minY maxX maxY', min_x, min_y, max_x, max_y)
    bb_x1_gt = min_x
    bb_y1_gt = min_y
    bb_x2_gt = max_x
    bb_y2_gt = max_y
    width_gt = max_y - min_y
    height_gt = max_x - min_x    
  else:
    img_rows, img_cols = input_img_prev.shape
    bb_x1_gt = 1
    bb_y1_gt = 1
    bb_x2_gt = img_rows
    bb_y2_gt = img_cols
    width_gt = img_cols
    height_gt = img_rows
  print ('width_gt height_gt', width_gt, height_gt)
  print ('bb_x1_gt, bb_y1_gt, bb_x2_gt, bb_y2_gt', bb_x1_gt, bb_y1_gt, bb_x2_gt, bb_y2_gt)
  # Choose current use bbox
  if ((flag_predict_use_SURF == False) and (flag_predict_use_Dense == False)) or (index < 2):
    bb_x1 = bb_x1_gt
    bb_y1 = bb_y1_gt
    bb_x2 = bb_x2_gt
    bb_y2 = bb_y2_gt
    width = width_gt
    height = height_gt
  else:
    bb_x1 = prev_bbx_x1
    bb_y1 = prev_bbx_y1
    bb_x2 = prev_bbx_x2
    bb_y2 = prev_bbx_y2
    width = bb_y2 - bb_y1
    height = bb_x2 - bb_x1

  #print ('bb', bb_x1, bb_y1, bb_x2, bb_y2)

  

  img_curr_rgb_clone = img_curr_rgb.copy()
  cv2.rectangle(img_curr_rgb_clone, (bb_x1, bb_y1), (bb_x2, bb_y2), (0, 255, 0), 2)  # Draw ground truth bbx  
  
  # create a CLAHE object (Arguments are optional).
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl1 = clahe.apply(input_img_prev)
  input_img_prev = cl1;

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl2 = clahe.apply(input_img_curr)
  input_img_curr = cl2;

  # ------ Save BBox (x1, y1, x2, y2) with (h, w)
  img_rows, img_cols = input_img_prev.shape
  scale = 0.3
  bbox_x1 = int(max(0, bb_x1 - scale*height)) #refPt[0][1]
  bbox_x2 = int(min(bb_x2 + scale*height, img_cols)) #refPt[1][1]
  bbox_y1 = int(max(0, bb_y1 - scale*width))#refPt[0][0]
  bbox_y2 = int(min(bb_y2 + scale*width, img_rows)) #refPt[1][0]
  refPt = np.empty([2,2])
  refPt[0][1] = bbox_x1
  refPt[1][1] = bbox_x2
  refPt[0][0] = bbox_y1
  refPt[1][0] = bbox_y2
  # print bbox_x1, bbox_x2, bbox_y1, bbox_y2
  height = bbox_x2 - bbox_x1
  width = bbox_y2 - bbox_y1

  print ('bbox', bbox_x1, bbox_x2, bbox_y1, bbox_y2)
  print ('bbox_width*height', width, height)

  cv2.rectangle(img_curr_rgb_clone, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 0, 255), 2)
  str_temp = 'Ground Truth'
  cv2.putText(img_curr_rgb_clone,str_temp,(10,30), font, 0.5,(0,255,0),2)
  str_temp = '| BBox Extend'
  cv2.putText(img_curr_rgb_clone,str_temp,(130,30), font, 0.5,(0,0,255),2)
  cv2.namedWindow('Ground Truth', cv2.WINDOW_AUTOSIZE)
  total_frame = len(ground_truth_array);
  current_frame_str = 'Frame: '
  current_frame_str += str(index)
  current_frame_str += ' / '
  current_frame_str += str(total_frame);
  print ('img_rows', img_rows)
  cv2.putText(img_curr_rgb_clone,current_frame_str,(10, int(img_rows - 20)), font, 0.5,(255,255,255),2)
  cv2.imshow('Ground Truth',img_curr_rgb_clone)
  cv2.moveWindow('Ground Truth', 100, 100)
  #cv2.waitKey(0)
  #print bbox_x1, bbox_y1, bbox_x2, bbox_y2, height, width
  input_img_prev = input_img_prev[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
  input_img_curr = input_img_curr[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
  
  #cv2.namedWindow('input_img_prev', cv2.WINDOW_AUTOSIZE)
  #cv2.imshow('input_img_prev',input_img_prev) 

  #cv2.namedWindow('input_img_curr', cv2.WINDOW_AUTOSIZE)
  #cv2.imshow('input_img_curr',input_img_curr)

  #print ('input_img_prev', input_img_prev.shape)
  


  # --------------------  SURF Points --------------- #
  surf = cv2.SURF(20)
  e1 = cv2.getTickCount()
  kp, des = surf.detectAndCompute(input_img_prev,None)
  e2 = cv2.getTickCount()
  time = (e2 - e1)/ cv2.getTickFrequency()
  print ('kp', len(kp))
  
  img_curr_rgb_predict = img_curr_rgb.copy()
  if (len(kp) > 0):
    img_prev_ROI_RGB = inputImg1[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
    input_img_prev_surf = cv2.drawKeypoints(img_prev_ROI_RGB,kp,None,(255,0,0),4)
    cv2.namedWindow('SURF Points', cv2.WINDOW_AUTOSIZE)
    img_prev_RGB = inputImg1.copy()
    str_temp = 'SURF Points in ROI: '
    str_temp += str(len(kp))
    str_temp += ' | Time: '
    str_temp += str(time)[:5]
    str_temp += ' s'
    cv2.putText(img_prev_RGB,str_temp,(10,30), font, 0.5,(255,255,255),2)
    img_prev_RGB[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = input_img_prev_surf

    overlay = img_prev_RGB.copy()
    #cv2.circle(overlay, (166, 132), 12, (255, 0, 0), -1)
    cv2.rectangle(overlay, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 0, 255), -1)
    opacity = 0.2 
    cv2.addWeighted(overlay, opacity, img_prev_RGB, 1 - opacity, 0, img_prev_RGB)  
    cv2.imshow('SURF Points',img_prev_RGB) 

    pt2 = []
    pt2_narray = np.empty((0,2), float)
    #print ('pt2_array', pt2_narray)

    # Generate SURF points array
    index_SURF= 0
    for each_point in kp:
        pt2.append(each_point.pt)
        x = each_point.pt[0]
        y = each_point.pt[1]
        pt2_narray = np.vstack((pt2_narray, np.array([x, y])))
        index_SURF = index_SURF + 1
    pt2_narray =  np.float32(pt2_narray).reshape(-1, 1, 2)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                         qualityLevel = 0.3,
                         minDistance = 7,
                         blockSize = 7 )
    p0 = pt2_narray;
    p1, status, err = cv2.calcOpticalFlowPyrLK(input_img_prev, input_img_curr, p0, None, **lk_params)
    print ('p1_type', type(p1))
    print ('p1_size', p1.shape)  # print ('p1', p1)

    p0r, status, err = cv2.calcOpticalFlowPyrLK(input_img_curr, input_img_prev, p1, None, **lk_params)
    print ('p0r_type', type(p1))
    print ('p0r_size', p1.shape)

    p0_array = squeeze_pts(p0)
    p0r_array = squeeze_pts(p0r)
    #p0r_array = np.array([p0r.reshape(-1, 2)]) #.reshape(-1, 2)
    #print p0r
    fb_err = np.sqrt(np.power(p0_array - p0r_array, 2).sum(axis=1))
    print ('fb_err.shape', fb_err.shape)  
    print ('fb_err', fb_err)

    good = fb_err < 1
    print (good)
    
    #d = abs(p0-p0r).reshape(-1, 2).max(-1)
    #good = d < 1  
    #print ('d.shape', d.shape)  
    #print ('d', d)
    

    print (good)
    good_curr_points = p1[status == 1]
    good_prev_points = p0[status == 1]
    # Create some random colors
    

    img_prev_ROI_RGB = inputImg1[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
    img_prev_ROI_RGB_all_points = img_prev_ROI_RGB.copy()

    img_curr_ROI_RGB = inputImg2[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
    img_curr_ROI_RGB_all_points = img_curr_ROI_RGB.copy()

    vis3_good = img_curr_ROI_RGB.copy()
    # vis4_all = inputImg2.copy()
    vis4_all = cv2.addWeighted(img_prev_ROI_RGB,0.5,img_curr_ROI_RGB,0.5,1)
    #print ('p1', p1)
    opacity = 0.6

    # ---------- Prediction BBox by SURF points-------
    predict_bbox_x1 = img_cols
    predict_bbox_y1 = img_rows
    predict_bbox_x2 = 0
    predict_bbox_y2 = 0
      
    for i,(new, old, good_flag) in enumerate(zip(good_curr_points, good_prev_points, good)): 
        a,b = new.ravel()
        img_curr_ROI_RGB_all_points_clone = img_curr_ROI_RGB_all_points.copy()
        cv2.circle(img_curr_ROI_RGB_all_points_clone, (a, b), 3, color[i].tolist(), -1)
        cv2.addWeighted(img_curr_ROI_RGB_all_points_clone, opacity, img_curr_ROI_RGB_all_points, 1 - opacity, 0, img_curr_ROI_RGB_all_points)
        c,d = old.ravel() 
        cv2.circle(img_prev_ROI_RGB_all_points, (c, d), 3, color[i].tolist(), 1)
        cv2.line(vis4_all, (a,b),(c,d), color[i].tolist(), 2)
        if not (good_flag):
            continue    
        cv2.line(vis3_good, (a,b),(c,d), color[i].tolist(), 2)
        if (predict_bbox_x1 > a): predict_bbox_x1 = a
        if (predict_bbox_x2 < a): predict_bbox_x2 = a
        if (predict_bbox_y1 > b): predict_bbox_y1 = b
        if (predict_bbox_y2 < b): predict_bbox_y2 = b
        
   
    

    surf_bbox_scale = 0.5
    surf_bbox_width = predict_bbox_y2 - predict_bbox_y1
    surf_bbox_height = predict_bbox_x2 - predict_bbox_x1
    predict_bbox_surf_x1 = int((predict_bbox_x1 + bbox_x1) - surf_bbox_scale * surf_bbox_height)
    predict_bbox_surf_y1 = int((predict_bbox_y1 + bbox_y1) - surf_bbox_scale * surf_bbox_width)
    predict_bbox_surf_x2 = int((predict_bbox_x2 + bbox_x1) + surf_bbox_scale * surf_bbox_height)
    predict_bbox_surf_y2 = int((predict_bbox_y2 + bbox_y1) + surf_bbox_scale * surf_bbox_width)
    


    # Copy img_prev_ROI_RGB_all_points (ROI with Interested Points) to cl1
    #Original_Img1 = cl1.copy()
    img_prev_RGB_all_points = inputImg1.copy()
    img_prev_RGB_all_points[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = img_prev_ROI_RGB_all_points

    cv2.namedWindow('Previous Image All Points', cv2.WINDOW_AUTOSIZE)
    str_temp = 'Prev. Img. | Good Points: '
    str_temp += str(len(good_prev_points))
    cv2.putText(img_prev_RGB_all_points,str_temp,(10,30), font, 0.5,(255,255,255),2)
    cv2.imshow('Previous Image All Points',img_prev_RGB_all_points) 

    # Copy img_prev_ROI_RGB_all_points (ROI with Interested Points) to cl2
    img_curr_RGB_all_points = inputImg2.copy()
    img_curr_RGB_all_points[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = img_curr_ROI_RGB_all_points
    cv2.namedWindow('Current Image All Points', cv2.WINDOW_AUTOSIZE)
    str_temp = 'Curr. Img. | Good Points: '
    str_temp += str(len(good_curr_points))
    cv2.putText(img_curr_RGB_all_points,str_temp,(10,30), font, 0.5,(255,255,255),2)
    cv2.imshow('Current Image All Points',img_curr_RGB_all_points)

    inputImg2_good_points_RGB = inputImg2.copy()
    inputImg2_good_points_RGB[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = vis3_good
    cv2.namedWindow('Matched Good Points', cv2.WINDOW_AUTOSIZE)
    str_temp = 'Matched Good Points: '
    str_temp += str(sum(good))
    cv2.putText(inputImg2_good_points_RGB,str_temp,(10,30), font, 0.5,(255,255,255),2)
    cv2.imshow('Matched Good Points',inputImg2_good_points_RGB) 

    inputImg2_all_points_RGB = inputImg2.copy()
    inputImg2_all_points_RGB[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = vis4_all
    cv2.namedWindow('All Matched Points', cv2.WINDOW_AUTOSIZE)
    str_temp = 'Matched All Points: '
    str_temp += str(len(good_curr_points))
    cv2.putText(inputImg2_all_points_RGB,str_temp,(10,30), font, 0.5,(255,255,255),2)
    cv2.imshow('All Matched Points',inputImg2_all_points_RGB)


  # ---------------- Dense Flow -------------------- #
  e1 = cv2.getTickCount()
  flow = cv2.calcOpticalFlowFarneback(input_img_prev,input_img_curr,0.5,1,3,15,3,5,1)
  e2 = cv2.getTickCount()
  time = (e2 - e1)/ cv2.getTickFrequency()

   # print ('flow', flow)
  print ('flow_size', flow.shape)
  mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

  # Select num_largest points by dense flow method
  img_prev_ROI_RGB = inputImg1[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
  dens_points_gray = np.zeros_like(img_prev_ROI_RGB)
  print ('mag length', len(mag))
  num_largest = min(450, len(mag))  
  indices = (-mag).argpartition(num_largest, axis=None)[:num_largest]
  x, y = np.unravel_index(indices, mag.shape)
  #print ('x=', x)
  #print ('y=', y)

  # ------------------ Predict BBox by Dense points -------------------
  predict_bbox_x1 = img_cols
  predict_bbox_y1 = img_rows
  predict_bbox_x2 = 0
  predict_bbox_y2 = 0
  for i in range(0, num_largest):
    cv2.circle(dens_points_gray, (y[i], x[i]), 3, color[i].tolist(), 1)
    if (predict_bbox_x1 > y[i]): predict_bbox_x1 = y[i]
    if (predict_bbox_x2 < y[i]): predict_bbox_x2 = y[i]
    if (predict_bbox_y1 > x[i]): predict_bbox_y1 = x[i]
    if (predict_bbox_y2 < x[i]): predict_bbox_y2 = x[i]

  dense_bbox_scale = 0.5
  dense_bbox_width = predict_bbox_y2 - predict_bbox_y1
  dense_bbox_height = predict_bbox_x2 - predict_bbox_x1
  predict_bbox_dense_x1 = int((predict_bbox_x1 + bbox_x1) - dense_bbox_scale * dense_bbox_height)
  predict_bbox_dense_y1 = int((predict_bbox_y1 + bbox_y1) - dense_bbox_scale * dense_bbox_width)
  predict_bbox_dense_x2 = int((predict_bbox_x2 + bbox_x1) + dense_bbox_scale * dense_bbox_height)
  predict_bbox_dense_y2 = int((predict_bbox_y2 + bbox_y1) + dense_bbox_scale * dense_bbox_width)
  
  # Draw all BBox
  str_temp = 'Ground Truth'
  cv2.putText(img_curr_rgb_predict,str_temp,(10,30), font, 0.5,(0,255,0),2)
  str_temp = '| BBox Extend'
  cv2.putText(img_curr_rgb_predict,str_temp,(130,30), font, 0.5,(0,0,255),2)
  str_temp = '| SURF Predict'
  cv2.putText(img_curr_rgb_predict,str_temp,(250,30), font, 0.5,(255,0,0),2)
  str_temp = '| Dense Predict'
  cv2.putText(img_curr_rgb_predict,str_temp,(370,30), font, 0.5,(255,255,0),2)
  if (len(kp) > 0): 
    cv2.rectangle(img_curr_rgb_predict, (predict_bbox_surf_x1, predict_bbox_surf_y1), (predict_bbox_surf_x2, predict_bbox_surf_y2), (255, 0, 0), 2)
  cv2.rectangle(img_curr_rgb_predict, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 0, 255), 2)
  cv2.rectangle(img_curr_rgb_predict, (bb_x1, bb_y1), (bb_x2, bb_y2), (0, 255, 0), 2)  # Draw ground truth bbx
  cv2.rectangle(img_curr_rgb_predict, (predict_bbox_dense_x1, predict_bbox_dense_y1), (predict_bbox_dense_x2, predict_bbox_dense_y2), (255, 255, 0), 2)
  cv2.namedWindow('img_curr_rgb_predict', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('img_curr_rgb_predict',img_curr_rgb_predict) 

  # Draw Dense Flow selected points
  img1_original_rgb = np.zeros_like(inputImg1)
  img1_original_rgb[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = dens_points_gray
  cv2.namedWindow('Dense Optical Flow', cv2.WINDOW_AUTOSIZE)
  str_temp = 'ROI Dense Optical Flow Selected '
  str_temp += str(num_largest)
  str_temp += ' Points'
  cv2.putText(img1_original_rgb,str_temp,(10,30), font, 0.5,(255,255,255),2)
  
  cv2.imshow('Dense Optical Flow',img1_original_rgb)

  # Draw HSV Flow Code
  img_prev_ROI_RGB = inputImg1[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
  hsv = np.zeros_like(img_prev_ROI_RGB)
  hsv[...,1] = 255
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

  print ('hsv_size', hsv.shape)
  print ('hsv_type', type(hsv))
  print ('inputImg1', inputImg1.shape)
  print ('inputImg1_type', type(inputImg1))
   
  rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
  img1_original_rgb = np.zeros_like(inputImg1)
  img1_original_rgb[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = rgb
  cv2.namedWindow('Optical Flow HSV', cv2.WINDOW_AUTOSIZE)
  str_temp = 'ROI Dense Optical Flow'
  str_temp += ' | Time: '
  str_temp += str(time)[:5]
  str_temp += ' s'
  cv2.putText(img1_original_rgb,str_temp,(10,30), font, 0.5,(255,255,255),2)
  cv2.imshow('Optical Flow HSV',img1_original_rgb)


  bbx_scale_damping = 0.8
  # Save prev bbox size
  if (flag_predict_use_SURF == True) and (index > 1):
    prev_bbx_x1 = int(prev_bbx_x1 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_surf_x1)
    prev_bbx_y1 = int(prev_bbx_y1 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_surf_y1)
    prev_bbx_x2 = int(prev_bbx_x2 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_surf_x2)
    prev_bbx_y2 = int(prev_bbx_y2 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_surf_y2)
  elif (flag_predict_use_SURF == True) and (index == 1):
    prev_bbx_x1 = predict_bbox_surf_x1
    prev_bbx_y1 = predict_bbox_surf_y1
    prev_bbx_x2 = predict_bbox_surf_x2
    prev_bbx_y2 = predict_bbox_surf_y2
  elif (flag_predict_use_Dense == True) and (index > 1):
    prev_bbx_x1 = prev_bbx_x1 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_dense_x1
    prev_bbx_y1 = prev_bbx_y1 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_dense_y1
    prev_bbx_x2 = prev_bbx_x2 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_dense_x2
    prev_bbx_y2 = prev_bbx_y2 * bbx_scale_damping + (1 - bbx_scale_damping) * predict_bbox_dense_y2
  elif (flag_predict_use_SURF == True) and (index == 1):
    prev_bbx_x1 = predict_bbox_dense_x1
    prev_bbx_y1 = predict_bbox_dense_y1
    prev_bbx_x2 = predict_bbox_dense_x2
    prev_bbx_y2 = predict_bbox_dense_y2


  # --- Test Morphological Filter -------------#
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

  prev_img_gray = cv2.cvtColor(inputImg1, cv2.COLOR_BGR2GRAY)
  curr_img_gray = cv2.cvtColor(inputImg2, cv2.COLOR_BGR2GRAY)

  print ('kernel', kernel)  
  img_closing = cv2.morphologyEx(curr_img_gray, cv2.MORPH_CLOSE, kernel)
  cv2.namedWindow('Current Closing', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('Current Closing',img_closing) 

  img_opening = cv2.morphologyEx(curr_img_gray, cv2.MORPH_OPEN, kernel)  
  cv2.namedWindow('Current Open', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('Current Open',img_opening)   

  img_minus = img_closing - img_opening;
  cv2.namedWindow('Current Close - Open', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('Current Close - Open',img_minus)   

  #abs_diff_img = prev_img_gray.copy()
  abs_diff_img = cv2.absdiff(curr_img_gray, prev_img_gray)
  cv2.namedWindow('Current Difference', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('Current Difference',abs_diff_img)   

  adaptive_img = cv2.adaptiveThreshold(img_minus,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)
  cv2.namedWindow('THRESH_OTSU', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('THRESH_OTSU',adaptive_img)   
  # param_w = 2
  # windos_size = 2*param_w + 1;
  # img_rows, img_cols = curr_img_gray.shape
  # filter_img = curr_img_gray.copy()
  
  # print ('row, cols',img_rows, img_cols)

  # for row_ind in range(int(math.floor(windos_size)), img_rows - int(math.floor(windos_size))):
  #   for column_ind in range(int(math.floor(windos_size)), img_cols - int(math.floor(windos_size))):
  #     max_1 = 0;      
  #     curr_pixel = curr_img_gray[row_ind, column_ind]      
  #     for max_i in range(-param_w, param_w):
  #       min_1 = 999;
  #       for min_j in range(-param_w, param_w):          
  #          if curr_img_gray[row_ind + max_i + min_j, column_ind] < min_1:
  #           min_1 = curr_img_gray[row_ind + max_i + min_j, column_ind]
  #       if min_1 > max_1:
  #         max_1 = min_1

  #     max_2 = 0;
  #     for max_i in range(-param_w, param_w):
  #       min_2 = 999;
  #       for min_j in range(-param_w, param_w):
  #         if curr_img_gray[row_ind + max_i + min_j, column_ind] < min_2:
  #           min_2 = curr_img_gray[row_ind + max_i + min_j, column_ind]
  #       if min_2 > max_2:
  #         max_2 = min_2
  #   curr_img_gray[row_ind, column_ind] = curr_pixel - max(max_1, max_2)

  # cv2.namedWindow('Current positive filter', cv2.WINDOW_AUTOSIZE)
  # cv2.imshow('Current positive filter',curr_img_gray)  

        # crop_x = column_ind - math.floor(windos_size/2);
        # crop_y = row_ind - math.floor(windos_size/2);
        # crop_img = curr_img_gray[crop_y:crop_y+windos_size-1, crop_x:crop_x+windos_size-1]



def get_image_sequence(image_folder):
  global clone, index
  img_directory = image_folder
  img_directory += '/%08d.jpg'
  source = cv2.VideoCapture(img_directory)
  index = 0
  flag_first_img = True
  while (cv2.waitKey() & 0xff) != ord('q'):  # Press 'q' to quit the loop
    #if index == 8: break  
    retval, img_curr_rgb = source.read()
    
    print index
    print retval
    if not retval:
      break
    #cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('img',img_curr_rgb)
    #cv2.waitKey(0)

    if flag_first_img == True:
      flag_first_img = False          
    else:
        image_process(img_last_rgb, img_curr_rgb)
    index = index + 1
    img_last_rgb = img_curr_rgb.copy()


def get_image_ground_truth(image_folder):
  global ground_truth_array
  #csv_ground_truth_array = []
  #original_filename = 'landing_data_3_left/groundtruth.txt'
  original_filename = image_folder
  original_filename += '/groundtruth.txt'
  print original_filename
  #test_file = open(original_filename, "r")
  #csv_reader = csv.reader(test_file, delimiter='\t')
  #for row in csv_reader:
  #  csv_ground_truth_array.append(row)
  ground_truth_array = np.genfromtxt(original_filename, dtype=float, delimiter=',') 
  #print ground_truth_array.shape
if __name__ == "__main__":
  # execute only if run as a script 
  image_folder = '151221_Fixed_Wing_Right_Select_2'# ' #
  #image_folder = 'landing_data_2_left'
  #image_folder = '151121_Fixed_Wing_Both_Right_Final'
  get_image_ground_truth(image_folder)
  get_image_sequence(image_folder) 
  cv2.waitKey(0)
  cv2.destroyAllWindows()
