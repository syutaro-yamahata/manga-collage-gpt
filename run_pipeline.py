import os
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO
from openai import AzureOpenAI   # ← ★これが抜けていた

from google.cloud import vision

def get_vision_client():
    # ここも from_service_account_json を使うように修正
    return vision.ImageAnnotatorClient.from_service_account_json("credentials.json")





# =========================================================
# 0. パス・基本設定
# =========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

IMAGE_PATH = os.path.join(BASE_DIR, "debug_ocr.png")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
BALLOON_DIR = os.path.join(BASE_DIR, "balloons")

os.makedirs(BALLOON_DIR, exist_ok=True)

# =========================================================
# 1. Azure OpenAI 準備
# =========================================================

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

if not endpoint or not api_key:
    raise RuntimeError("Azure OpenAI の環境変数が設定されていません")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key,
)

def refine_ocr_text_with_gpt(texts: list[str]) -> str:
    """
    OCR結果（複数）→ 自然な漫画セリフ1文に整形
    """
    joined = " ".join(texts)

    prompt = f"""
以下は漫画の吹き出しをOCRした結果です。
誤認識・重複・ノイズを除去し、
自然な漫画のセリフ1文にしてください。

OCR結果:
{joined}
"""

    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "あなたは漫画編集者です。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=100,
    )

    return resp.choices[0].message.content.strip()

# =========================================================
# 2. YOLOで吹き出し検出
# =========================================================

print("🚀 YOLOで吹き出し検出中...")

if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLOモデルが見つかりません: {YOLO_MODEL_PATH}")

model = YOLO(YOLO_MODEL_PATH)
results = model(IMAGE_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("画像の読み込みに失敗しました")

balloons = []
idx = 0

for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < 0.4:
            continue  # 低信頼度は捨てる

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        out_path = os.path.join(BALLOON_DIR, f"balloon_{idx}.png")
        cv2.imwrite(out_path, crop)
        balloons.append(out_path)
        idx += 1

print(f"✅ 吹き出し数: {len(balloons)}")

# =========================================================
# 3. MMOCR（遅延 import）
# =========================================================

# print("\n🔍 MMOCRで文字認識中...")

# from mmocr.apis import MMOCRInferencer

# ocr = MMOCRInferencer(
#     det="dbnetpp",
#     rec="sar",
#     device="cpu"  # Windows安定優先
# )

# ocr_results = []

# for path in balloons:
#     out = ocr(path)

#     if not out["predictions"]:
#         ocr_results.append([])
#         continue

#     pred = out["predictions"][0]
#     texts = [
#         t for t, s in zip(pred["rec_texts"], pred["rec_scores"])
#         if s >= 0.4
#     ]
#     ocr_results.append(texts)

# =========================================================
# 4. GPTでセリフ整形
# =========================================================

ocr_results = []

print("\n✍️ GPTでセリフ整形\n")

for i, texts in enumerate(ocr_results):
    print(f"🗨 balloon_{i}.png")

    if not texts:
        print("   （文字なし）\n")
        continue

    line = refine_ocr_text_with_gpt(texts)
    print(" 👉", line, "\n")

print("🎉 完了")

def run_mmocr(balloon_images):
    # ★ 遅延 import（超重要）
    from mmocr.apis import MMOCRInferencer

    ocr = MMOCRInferencer(
        det="dbnetpp",
        rec="sar",
        device="cpu"
    )

    results = []
    for path in balloon_images:
        out = ocr(path)
        pred = out["predictions"][0]
        texts = [
            t for t, s in zip(pred["rec_texts"], pred["rec_scores"])
            if s >= 0.4
        ]
        results.append(texts)

    return results
