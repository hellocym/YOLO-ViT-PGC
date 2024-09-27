# Object Detecion 
import cv2
from ultralytics import YOLO
#plots
import matplotlib.pyplot as plt
import seaborn as sns

#basics
import pandas as pd
import numpy as np
import os
import subprocess

from tqdm import tqdm

cam1_path = '/hy-tmp/七贤岭/417290'
cam2_path = '/hy-tmp/七贤岭/417334'
# path = './行人提取的视频/qx242_231208_071059_071200.mp4'

#loading a YOLO model 
model = YOLO('yolov8x.pt')

#geting names from classes
dict_classes = model.model.names

### Configurations
# Scaling percentage of original frame
scale_percent = 100
#-------------------------------------------------------
# Reading video with cv2
video = cv2.VideoCapture(path)

# Objects to detect Yolo
class_IDS = [0] 
# Auxiliary variables
centers_old = {}


# Original informations of video
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video.get(cv2.CAP_PROP_FPS)
print('[INFO] - Original Dim: ', (width, height))

# Scaling Video for better performance 
if scale_percent != 100:
    print('[INFO] - Scaling change may cause errors in pixels lines ')
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    print('[INFO] - Dim Scaled: ', (width, height))
    
#-------------------------------------------------------
### Video output ####
video_name = 'track_result2.mp4'
output_path = "rep_" + video_name
tmp_output_path = "tmp_" + output_path
VIDEO_CODEC = "MP4V"

output_video = cv2.VideoWriter(tmp_output_path, 
                               cv2.VideoWriter_fourcc(*VIDEO_CODEC), 
                               fps, (width, height))
#-------------------------------------------------------

#loading a YOLO model 
model = YOLO('yolov8x.pt')


results = model.track(path, classes=0, persist=True)

for r in results:
    frame = r.plot()
    # add number of people in the frame, left bottom corner, red color, white text, large font
    cur_cnt = r.boxes.id.shape[0]
    cv2.putText(frame, f'People: {cur_cnt}', (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 10, 25)
    output_video.write(frame)
output_video.release()