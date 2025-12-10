# PIL（画像処理ライブラリ）とOS操作関連モジュールのインポート
from PIL import Image, ImageDraw, ImageFont
import os, io, cv2, sys
from google.cloud import vision  # Google Cloud Vision API のクライアント
import pykakasi  # ふりがな生成

# FuriganaDetection を読み込み


sys.path.append(os.path.abspath("."))  # プロジェクトルートを追加

from FuriganaDetection.src.detection import FuriganaDetector



# --- 検出器を初期化 ---
detector = FuriganaDetector(weights="FuriganaDetection/weights/furigana_model.pth")

# 縦書き文字描画関数（吹き出し高さに応じて均等配置）
def draw_vertical_text(draw, position, text, font, box_height, fill=(0,0,0)):
    n = len(text)
    spacing = (box_height - n * font.size) // (n - 1) if n > 1 else 0
    total_height = n * font.size + (n - 1) * spacing
    x, y = position
    y = y + (box_height - total_height) // 2
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        y += font.size + spacing

# 縦書き＋ルビ描画
def draw_vertical_text_with_ruby(draw, position, text, font, ruby_font, box_height, fill=(0,0,0)):
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    n = len(result)
    spacing = (box_height - n * font.size) // (n - 1) if n > 1 else 0
    total_height = n * font.size + (n - 1) * spacing
    x, y = position
    y = y + (box_height - total_height) // 2

    for item in result:
        ch = item['orig']
        ruby = item['hira']
        # 本文
        draw.text((x, y), ch, font=font, fill=fill)
        # ルビ（漢字の右上に小さく）
        rw, rh = draw.textsize(ruby, font=ruby_font)
        draw.text((x + font.size, y - rh//2), ruby, font=ruby_font, fill=fill)
        y += font.size + spacing

# 画像ファイルと置換ペアを元に画像内の文字を編集するメイン関数
def process_image(image_path, replacements):

    # 入力画像をRGBA（透明度付き）で開く
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # OpenCV形式で読み込み（ふりがな検出用）
    cv_img = cv2.imread(image_path)

    # --- ① FuriganaDetectionでふりがな削除 ---
    boxes = detector.detect(cv_img)
    for (x0, y0, x1, y1) in boxes:
        draw.rectangle([x0, y0, x1, y1], fill="white")

    # --- ② Google OCRで本文領域を検出 ---
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image) 
    texts = response.text_annotations  

    if texts:
        print("全文：", texts[0].description)

        # ③ OCR結果を走査して置換
        for pair in replacements:
            old_text = pair['from']
            new_text = pair['to']
            found = False

            for text in texts[1:]:
                if old_text in text.description:
                    vertices = text.bounding_poly.vertices
                    x_min = min([v.x for v in vertices])
                    y_min = min([v.y for v in vertices])
                    x_max = max([v.x for v in vertices])
                    y_max = max([v.y for v in vertices])
                    w, h = x_max - x_min, y_max - y_min

                    # --- 背景を白で塗りつぶす（ふりがなごと削除） ---
                    padding = 5
                    draw.rectangle([x_min - padding, y_min - padding,
                                    x_max + padding, y_max + padding],
                                    fill="white")

                    # フォント定義
                    try:
                        main_font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", int(h*0.8))
                        ruby_font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", int(h*0.3))
                    except:
                        main_font = ImageFont.load_default()
                        ruby_font = ImageFont.load_default()

                    # --- ④ 新しい文字を描画（縦横判定 + ルビ付き） ---
                    if h > w:
                        # 縦長 → 縦書き（ルビ付き）
                        draw_vertical_text_with_ruby(draw, (x_min + (w - main_font.size)//2, y_min),
                                                     new_text, main_font, ruby_font, h)
                    else:
                        # 横長 → 横書き（簡易ルビ対応）
                        draw.text((x_min, y_min), new_text, font=main_font, fill="black")
                        kks = pykakasi.kakasi()
                        result = kks.convert(new_text)
                        rx, ry = x_min, y_min - ruby_font.size
                        for item in result:
                            draw.text((rx, ry), item['hira'], font=ruby_font, fill="black")
                            rw, _ = draw.textsize(item['orig'], font=main_font)
                            rx += rw

                    found = True

            if not found:
                print(f"「{old_text}」が見つかりませんでした")
    else:
        print("OCRで文字を検出できませんでした")

    # ⑤ 保存
    base, ext = os.path.splitext(image_path)
    result_path = f"{base}_replaced.png"
    img.save(result_path)
    print("保存しました:", result_path)
    return result_path
