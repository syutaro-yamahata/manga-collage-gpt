from ultralytics import YOLO
import cv2

# ① YOLOv8 の標準モデル（今は仮）
model = YOLO("yolov8n.pt")  # ← 後で吹き出し専用モデルに差し替える

img_path = "/mnt/c/Users/shuta/manga-collage-gpt/debug_ocr.png"

# ② 推論
results = model(img_path, conf=0.25)

# ③ 可視化
img = cv2.imread(img_path)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            img,
            f"{conf:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

cv2.imwrite("debug_yolo_boxes.png", img)
print("saved: debug_yolo_boxes.png")
