import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO
from google.cloud import vision
import io

# クライアント初期化
client = vision.ImageAnnotatorClient.from_service_account_json("credentials.json")
YOLO_MODEL_PATH = "models/best.pt"
model = YOLO(YOLO_MODEL_PATH)

def detect_text_google(cv_roi):
    """切り出した吹き出し内の文字を読み取り、順序を整える"""
    success, encoded_image = cv2.imencode('.png', cv_roi)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    
    response = client.document_text_detection(image=image)
    
    if response.full_text_annotation:
        return response.full_text_annotation.text.replace('\n', '')
    return ""

def draw_text_auto_direction(draw, text, x1, y1, x2, y2, font):
    """縦横比を判定して縦書きまたは横書きで描画する"""
    w = x2 - x1
    h = y2 - y1
    is_vertical = h > w * 1.1
    
    if is_vertical:
        char_size = font.size
        line_spacing = 4
        start_x = x1 + (w - char_size) // 2
        current_y = y1 + (h - (len(text) * (char_size + line_spacing))) // 2
        
        for char in text:
            if char == "ー": char = "｜"
            if char == "…": char = "："
            draw.text((start_x, current_y), char, font=font, fill="black")
            current_y += char_size + line_spacing
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw = right - left
        th = bottom - top
        cx = x1 + (w - tw) // 2
        cy = y1 + (h - th) // 2
        draw.text((cx, cy), text, font=font, fill="black")

def fill_all_speech_balloons_white(image_path, output_path, replacements):
    cv_img = cv2.imread(image_path)
    if cv_img is None: return image_path
    
    results = model(image_path)[0]
    detected_texts = []

    # --- ステップ1: 読み取りと消去 ---
    for box in results.boxes:
        if box.conf[0] < 0.4: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        roi = cv_img[y1:y2, x1:x2]
        
        # OCR実行（文字を描画しない場合でも、解析用に残しています）
        original_text = detect_text_google(roi)
        detected_texts.append(original_text)

        # 文字消去（中心部保護版）
        w, h = x2 - x1, y2 - y1
        px, py = int(w * 0.15), int(h * 0.15)
        center_roi = cv_img[y1+py : y2-py, x1+px : x2-px]
        if center_roi.size > 0:
            gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            center_roi[mask > 0] = [255, 255, 255]
            cv_img[y1+py : y2-py, x1+px : x2-px] = center_roi

    # PILへ変換
    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # =========================================================
    # ステップ2: 文字描画（いつでも戻せるようにコメントアウト中）
    # =========================================================
    """ 
    # 文字を入れたい時は、以下の「'''」と末尾の「'''」を削除してください
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 20)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(results.boxes):
        if box.conf[0] < 0.4: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if replacements and i < len(replacements):
            display_text = replacements[i].get("to", "...")
        elif i < len(detected_texts) and detected_texts[i]:
            display_text = detected_texts[i]
        else:
            display_text = "..."

        draw_text_auto_direction(draw, display_text, x1, y1, x2, y2, font)
    """
    # =========================================================

    img.save(output_path)
    return output_path