from ultralytics import YOLO
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from distance_calculation import DistanceCalculation
from sklearn.metrics import pairwise_distances

def resolve_lost_tracks(
    lost_tracks_prev: dict, lost_tracks_next: dict, metric: str = "manhattan"
):
    prev_id_to_next_id = dict()

    if metric == "cosine":
        similarity_matrix = cosine_similarity(
            list(lost_tracks_prev.values()), list(lost_tracks_next.values())
        )
        threshold = 0.9999
    elif metric == "manhattan":
        similarity_matrix = np.negative(
            pairwise_distances(
                list(lost_tracks_prev.values()),
                list(lost_tracks_next.values()),
                metric="manhattan",
            )
        )
        threshold = -50

    else:
        raise ValueError("Input must be either 'cosine' or 'manhattan'")

    while not np.all(np.isinf(similarity_matrix)):
        max_index = np.argmax(similarity_matrix)
        max_row, max_col = np.unravel_index(max_index, similarity_matrix.shape)
        max_value = similarity_matrix[max_row, max_col]

        if max_value > threshold:
            prev_id_to_next_id[list(lost_tracks_prev.keys())[max_row]] = list(
                lost_tracks_next.keys()
            )[max_col]
        else:
            return prev_id_to_next_id
        similarity_matrix[max_row, :] = -np.inf
        similarity_matrix[:, max_col] = -np.inf

    return prev_id_to_next_id


def augment_and_resize_frame(frame, augment: bool = False):
    if augment:
        gamma = 0.5
        frame = cv2.pow(frame / 255.0, gamma)
        frame = (frame * 255).astype("uint8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.8  # Increase saturation
        hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180  # Increase hue by 10 degrees
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    frame = cv2.resize(frame, (640, 640))

    return frame


def adam_tracking_v2(input_video_path: str, augment: bool=False, verbose: bool=True):
    
    print(f'Beginning tracking on {input_video_path}')

    # Defining distance calculation object
    distance_calculator = DistanceCalculation()

    # Loading object detection model
    model = YOLO('best.pt')

    # Reading input video
    input_video = cv2.VideoCapture(input_video_path)

    # Preparing output video parameters
    vidcap = cv2.VideoCapture(input_video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_frame_predictions = {}
    distances = {}

    for frame_number in range(1, total_frames):

        input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        _, original_next_frame = input_video.read()

        next_frame = augment_and_resize_frame(original_next_frame, augment=augment)

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
            if id not in distances:
                distances[id] = 0
            else:
                distances[id] += distance_calculator.cal_distance(
                    prev_frame_predictions[id], next_frame_predictions[id]
                )

        if verbose:
            print("Processed frame", frame_number, "of", total_frames)

    