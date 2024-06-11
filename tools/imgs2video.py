#!/usr/local/bin/python3

import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
ap.add_argument("-fps", "--frames_per_second", required=False, type=int, default=2, help="frame rate for the output video")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/show_bbox_n/leftImg8bit/val/lindau'
ext = args['extension']
output = args['output']
fps = args['frames_per_second']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# Sort the images to maintain a proper sequence
images.sort()

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video', frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, fps, (width, height))

for image in images:
    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)
    
    # Add the image file name text on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2
    text_position = (10, height - 10)  # Bottom-left corner
    
    cv2.putText(frame, image, text_position, font, font_scale, font_color, line_type)

    out.write(frame) # Write out frame to video

    cv2.imshow('video', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
