import cv
import sys
import cv2
import numpy as np
import csv
import math
import time
#import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage import exposure
from matplotlib import gridspec

import morphsnakes
from matplotlib import pyplot as ppl
import os


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
  input_img_prev = cl1.copy();

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl2 = clahe.apply(input_img_curr)
  input_img_curr = cl2.copy();

  # ------ Save BBox (x1, y1, x2, y2) with (h, w)
  img_rows, img_cols = input_img_prev.shape
  scale = 0.2
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

  # ------------------ Active Countour -----------

  #-----Splitting the LAB image to different channels-------------------------
  img = img_curr_rgb.copy()
  
  lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  # cv2.imshow('l_channel', l)
  # cv2.imshow('a_channel', a)
  # cv2.imshow('b_channel', b)

  #-----Applying CLAHE to L-channel-------------------------------------------
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  cl = clahe.apply(l)
  cv2.imshow('CLAHE output', cl)

  #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
  limg = cv2.merge((cl,a,b))

  img_countour = input_img_curr.copy()
  center_y = (bbox_y2 - bbox_y1)/2;
  center_x = (bbox_x2 - bbox_x1)/2;
  img_countour = img_countour[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
  # img_countour = img_countour[center_y-10:center_y+10, center_x-10:center_x+10]

  # img_countour = exposure.equalize_hist(img_countour)    
 

  img_countour_roi  = img_countour.copy()
  cv2.namedWindow(' ROI', cv2.WINDOW_AUTOSIZE)
  cv2.imshow(' ROI',img_countour_roi)

  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(30,30))

  img_closing = cv2.morphologyEx(img_countour_roi, cv2.MORPH_CLOSE, kernel)
  img_opening = cv2.morphologyEx(img_countour_roi, cv2.MORPH_OPEN, kernel)
  img_minus = img_closing - img_opening;
  cv2.namedWindow('Active_Countour_Minus', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('Active_Countour_Minus',img_minus)
  
  # g(I)
  # gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)
 
  gI = morphsnakes.gborders(img_minus, alpha=500, sigma=1)
  cv2.namedWindow('GI', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('GI',gI)


  # Morphological GAC. Initialization of the level-set.
  mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-1)
  # mgac.levelset = circle_levelset(img.shape, (159, 103), 50, scalerow=0.1) #
  rect_level_set = np.zeros(img_countour.shape)
 
  # # rect_level_set[243-10:276-10, 228+10:366+10] = 1
  [img_height,img_width] = img_countour.shape
  
  rect_level_set[3:img_height-3, 3:img_width-3] = 0.5
  # rect_level_set[center_y-10:center_y+10, center_x-10:center_x+10] = 1
  
  mgac.levelset = rect_level_set

   
  
  start_time = time.time()
  return_level_set = morphsnakes.evolve_visual(mgac, num_iters=10 , background=img_countour)
  delta_time_contour = (time.time() - start_time)
  print("--- %s seconds ---" % delta_time_contour)

  # print('return_level_set',return_level_set)
  print('return_level_set',return_level_set.shape)

  img_draw_contour = img_countour.copy()
  # --- Using python plot show ----
  # ppl.figure(1)
  # fig = ppl.gcf()
  # fig.clf()
  # gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 

  # ax1 = ppl.subplot(1,2,2)
  # ax1.imshow(img_draw_contour, cmap=ppl.cm.gray)
  # ax1.contour(return_level_set, [0.5], colors='r')
  # fig.canvas.draw()
  # ax2 = ppl.subplot(1,2,1)
  # ax2.imshow(input_img_curr, cmap=ppl.cm.gray) 
  # ppl.pause(0.05)

  return_level_set = cv2.resize(return_level_set,(160, 160), interpolation = cv2.INTER_CUBIC)
  contours, hierarchy = cv2.findContours(cv2.convertScaleAbs(return_level_set),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  img_curr_rgb_draw = img_curr_rgb.copy()
  img_curr_rgb_draw = img_curr_rgb_draw[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
  img_curr_rgb_draw = cv2.resize(img_curr_rgb_draw,(160, 160), interpolation = cv2.INTER_CUBIC)


  cv2.drawContours(img_curr_rgb_draw, contours, -1, (0,0,250), 2)
  cv2.namedWindow('contour', cv2.WINDOW_NORMAL)

  cv2.imshow('contour',img_curr_rgb_draw)
  cv2.resizeWindow('contour', 160,160)

  file_name = image_folder
  file_name += '_Contour_'
  file_name += str(index).zfill(3)
  file_name += '.jpg'
  cv2.imwrite(os.path.join(dirname, file_name), img_curr_rgb_draw)
  time_consume[index] = delta_time_contour

def get_image_sequence(image_folder):
  global clone, index
  img_directory = image_folder
  img_directory += '/%08d.jpg'
  source = cv2.VideoCapture(img_directory)
  index = 0
  flag_first_img = True
  # while (cv2.waitKey() & 0xff) != ord('q'):  # Press 'q' to quit the loop
  while 1:
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
  file_name = image_folder
  file_name += '_Contour_time_consume.txt'  
  np.savetxt(file_name, time_consume, delimiter=',',fmt='%.3f')   # X is an array


def get_image_ground_truth(image_folder):
  global ground_truth_array, time_consume
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
  [sum_frames, tmp]= ground_truth_array.shape
  time_consume = np.zeros(sum_frames)
 
if __name__ == "__main__":
  global image_folder, dirname
  # execute only if run as a script 
  image_folder = '151221_Fixed_Wing_Right_Select_2'# ' #
  # image_folder = 'landing_data_2_left'
  # image_folder = 'landing_data_1_left'
  # image_folder = '151121_Fixed_Wing_Both_Right_Final'
  # dirname = '151121_Fixed_Wing_Both_Right_Final_Contour'
  dirname = '151221_Fixed_Wing_Right_Select_2_Contour'
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  get_image_ground_truth(image_folder)
  get_image_sequence(image_folder) 
  cv2.waitKey(0)
  cv2.destroyAllWindows()
