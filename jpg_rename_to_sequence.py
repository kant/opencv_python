import os, os.path, time
import errno
import shutil

count_num = 0 # Set the image sequence from 0
src_dir = os.curdir # Set the source directory
dst_dir = os.path.join(os.curdir, "resutls") # Set the destination directory

if not os.path.exists(dst_dir):  # Make dst_dir directory for renamed files
	try:
		os.makedirs(dst_dir)
	except OSError as exception:  # For race-condition check
		if exception.errno != errno.EEXIST:
			raise

for file in os.listdir("."):  # Get all files in the current directory
    if file.endswith(".jpg"):  # Filter the files with ".tif" suufix of the file
     	files_base, file_extension = os.path.splitext(file) # Get base and extension of the current file
 
    	count_num += 1 # Sequence update
    	dst_file_name = "%08d"%(count_num)  # Set the renamed style with 00000001

    	dst = os.path.join(dst_dir, dst_file_name + file_extension) # set new dst with "dst_dir/dst_file_name.file_extension"
    	print dst   	
        print str(file) + " - Creation date: " + str(time.ctime(os.path.getctime(file)))
        shutil.copyfile(file, dst)

