import cv2

def apply_augmentation(frame):
    # Change brightness
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Increase brightness

    # Change exposure (gamma correction)
    gamma = 0.9
    frame = cv2.pow(frame / 255.0, gamma)
    frame = (frame * 255).astype('uint8')

    # Convert to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Apply CLAHE to Y channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

    # Convert back to BGR
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Change saturation and hue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation
    hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180  # Increase hue by 10 degrees
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame


# Open video file
video_capture = cv2.VideoCapture('sheeps_30sec.mp4')

# Get video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
output_video = cv2.VideoWriter('output_video1.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Apply augmentation
    augmented_frame = apply_augmentation(frame)

    # Write augmented frame to output video
    output_video.write(augmented_frame)

    cv2.imshow('Augmented Video', augmented_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
