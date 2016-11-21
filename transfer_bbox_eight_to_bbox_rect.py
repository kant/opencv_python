
import cv
import sys
import cv2
import numpy as np
import csv
import os, os.path, time
import datetime

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
  #ground_truth_array = csv.reader(original_filename, delimiter=',')
  #for row in ground_truth_array:
    #print row

def save_bbox_in_rect_format(image_folder):
  target_filename = image_folder
  target_filename += "/groundtruth_rect.txt"
  if os.path.isfile(target_filename):
    print ("groundtruth_rect.txt exist!")
    print ("Save to another file with timestamp!")
    target_filename = image_folder
    target_filename += "/groundtruth_rect_"
    target_filename += datetime.datetime.now().strftime('%Y%m%d%I%M%s')
    target_filename += ".txt"

  save_file= csv.writer(open(target_filename,'wb'))
  #print ground_truth_array
  #print len(ground_truth_array)
  for index in xrange(len(ground_truth_array)):
    #print ('ground_truth_array', type(ground_truth_array))
    #ground_truth_array = ground_truth_array
    #for index, (x1, y1, x2, y2, x3, y3, x4, y4) in np.ndenumerate(ground_truth_array):
    #print index
    #print ground_truth_array[0][0]
    x1 = float(ground_truth_array[index][0])
    y1 = float(ground_truth_array[index][1])
    x2 = float(ground_truth_array[index][2])
    y2 = float(ground_truth_array[index][3])
    x3 = float(ground_truth_array[index][4])
    y3 = float(ground_truth_array[index][5])
    x4 = float(ground_truth_array[index][6])
    y4 = float(ground_truth_array[index][7])   
    min_x = min(x1, x2, x3, x4)
    min_y = min(y1, y2, y3, y4)
    max_x = max(x1, x2, x3, x4)
    max_y = max(y1, y2, y3, y4)
    #print ('ground_truth_array', ground_truth_array[index-1])
    #print ('eight variables', x1, y1, x2, y2, x3, y3, x4, y4)
    #print ('minX minY maxX maxY', min_x, min_y, max_x, max_y)
    bb_x1_gt = min_x
    bb_y1_gt = min_y
    bb_x2_gt = max_x
    bb_y2_gt = max_y
    
    width_gt = max_x - min_x
    height_gt = max_y - min_y

    bb_center_x = bb_x1_gt + width_gt/2
    bb_center_y = bb_y1_gt + height_gt/2

    #print('bb_x1_gt, bb_y1_gt, width_gt, height_gt',bb_x1_gt, bb_y1_gt, width_gt, height_gt)
    save_file.writerow([str(bb_x1_gt), str(bb_y1_gt), str(width_gt), str(height_gt)])


if __name__ == "__main__":
  # execute only if run as a script 
  image_folder = './landing_data_1_left' #landing_data_2_left
  get_image_ground_truth(image_folder)
  save_bbox_in_rect_format(image_folder)
  #get_image_sequence(image_folder) 
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
