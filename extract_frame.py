import cv2
import os
import numpy as np

def extract_frames(video_path, max_frames=16, normalize=True):
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Video has 0 frames: {video_path}")
        return []

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=np.int32)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            # we normalize pixel values to [0, 1]
            frame = frame.astype(np.float32)/ 255.0
            frames.append(frame)


    # fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frame_count / fps

    # print(f"FPS: {fps}")
    # print(f"Total frames: {frame_count}")
    # print(f"Duration (s): {duration}")
    cap.release()
    return np.array(frames)
