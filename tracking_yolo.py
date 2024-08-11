import cv2
import ultralytics
from ultralytics import YOLO

model = YOLO('../MAR8/weights/best.pt')
model_old = YOLO('../FEB29/weights/yolov8_best.pt')
# results = model.track(source='sheeps_30sec.mp4', show=True, save=True, project='./result' , tracker="bytetrack.yaml", mode="track", save_txt=True, conf=0.5)

# results = model_old.track(source='sheeps_30sec.mp4', show=True, project='./result' , tracker="bytetrack.yaml", mode="track", conf=0.4)
# results = model.track(source='sheeps_30sec.mp4', show=True, project='./result' , tracker="bytetrack.yaml", mode="track", conf=0.5)

results = model.track(source='output_video5.avi', show=True, project='./result' , tracker="botsort.yaml", mode="track", conf=0.5, persist=True, save=True, save_txt=True)