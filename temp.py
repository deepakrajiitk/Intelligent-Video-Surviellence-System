import cv2
from datetime import timedelta

cap=cv2.VideoCapture("video_31.avi")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 18020;
td = timedelta(seconds=(frame_count / fps))
print(td)