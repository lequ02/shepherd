import cv2
import ultralytics
from ultralytics import YOLO

model = YOLO('../MAR8/weights/best.pt')
results = model.track(source='output_video4.avi', show=True, project='./result' , tracker="botsort.yaml", mode="track", conf=0.5, persist=True, save=True, save_txt=True)