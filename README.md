
# ***<u>SHEPHERD***</u>: An Automated AI Livestock Health Monitoring System

## About
SHEPHERD is the AI Livestock Health Monitoring System that helps local farmers in the Midwest region to monitor their livestock 24/7 and warns them of any potential health problems. Think of a "Fitbit", but for sheep. Instead of equipping each sheep with their own expensive and annoying monitor, we can leverage AI and preexisting camera systems to devliver a simple and cost-effective software solution to sheep farmers. Like humans, when sheep are feeling ill, they tend not to move much. SHEPHERD uses a special custom tracking algorithm to monitor the animals, track the distance they have moved, and warns the farmers if a particular sheep hasn't been active. SHEPHERD can detect the early warning signs of illness and flag those individual sheep at the most risk, saving farmers time and money and keeping our sheep healthy and happy!

SHEPHERD also won 1st place in a Regional AI Innovation Competition. 

## DEMO
https://github.com/user-attachments/assets/e6af79cf-8b9d-49b5-b155-b47297b31396


https://github.com/user-attachments/assets/2498d61e-3aa9-4abe-9950-67f65498135b




## Quick Start

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

