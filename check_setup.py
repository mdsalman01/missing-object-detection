import cv2, numpy, scipy, torch
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("GPU available:", torch.cuda.is_available())
model = YOLO("yolov8m.pt")
print(len(model.names), "classes")
print(model.names)