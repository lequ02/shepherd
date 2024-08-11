import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import ultralytics
from ultralytics import YOLO
from distance_calculation import DistanceCalculation
from sheep_statistic import SheepStatistic
import time
from helper_function import resolve_lost_tracks, augment_and_resize_frame

app = Flask(__name__)
CORS(app)
my_statistic = SheepStatistic()

@app.route('/upload_video', methods=['POST'])
def tracking():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    distance_calculator = DistanceCalculation()
    video = cv2.VideoCapture(file.filename) 
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO('best.pt')

    prev_frame_predictions = {}

    for frame_number in range(1, total_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        _, original_next_frame = video.read()

        next_frame = augment_and_resize_frame(original_next_frame, augment=True)

        results = model.track(
            source=next_frame,
            persist=True,
            tracker="botsort.yaml",
            mode="track",
            conf=0.5,
            verbose=False,
        )

        next_frame_predictions = dict(
            zip(
                results[0].boxes.id.int().tolist(), results[0].boxes.xyxy.int().tolist()
            )
        )

        next_frame_predictions = {
            k: next_frame_predictions[k] for k in sorted(next_frame_predictions)
        }

        lost_tracks_next = {
            id: coords for id, coords in next_frame_predictions.items() if id > 32
        }
        lost_tracks_prev = {
            id: coords
            for id, coords in prev_frame_predictions.items()
            if id not in lost_tracks_next.keys()
        }

        if lost_tracks_next and lost_tracks_prev:
            prev_id_to_next_id = resolve_lost_tracks(lost_tracks_prev, lost_tracks_next)
            for prev_id, next_id in prev_id_to_next_id.items():
                next_frame_predictions[prev_id] = next_frame_predictions.pop(next_id)

        for id in list(
            set(next_frame_predictions.keys()).intersection(
                set(prev_frame_predictions.keys())
            )
        ):
            if id not in my_statistic.get_current_statistic():
                my_statistic.update_statistic(id, 0)
            else:
                my_statistic.update_statistic(id, distance_calculator.cal_distance(
                    prev_frame_predictions[id], next_frame_predictions[id]
                ))
        prev_frame_predictions = next_frame_predictions
    my_statistic.reset()
    return  jsonify({'statistic_data': my_statistic.get_moved_distance()}), 200

@app.route('/get_current_statistic')
def get_current_statistic():
    return  jsonify({'statistic_data': my_statistic.get_statistic()}), 200