README:

HOW TO RECREATE TRACKING VIDEO:

	step1:
augment 30s video using augmentation2.py

	step2:
run obj tracking YOLO (tracking_yolo.py) on augmented video and save the results in txt files

	step3:
change the format of result txts from xywhn to xyxy using convert_format.py

	step4:
correct the IDs that were lost track during the process using id_correcting2.py

	step5:
draw the new bounding boxes using drawingboxes.py