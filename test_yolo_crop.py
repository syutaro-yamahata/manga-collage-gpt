from ultralytics import YOLO
import cv2
import os

model = YOLO("models/balloon.pt")

img_path = "debug_ocr.png"
img = cv2.imread(img_path)

results = model(img_path, conf=0.4)

os.makedirs("balloons", exist_ok=True)

count = 0
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        out = f"balloons/balloon_{count}.png"
        cv2.imwrite(out, crop)
        print("saved:", out)
        count += 1

print("total balloons:", count)
