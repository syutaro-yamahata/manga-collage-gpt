# Flask本体と各種ユーティリティ
from flask import Flask, render_template, request
import os
import sys
import uuid

# パス調整
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
from collage import fill_all_speech_balloons_white
from gpt_helper import parse_instruction_with_gpt

# 環境変数読み込み
load_dotenv()

app = Flask(__name__)

# ディレクトリ
UPLOAD_DIR = os.path.join("static", "uploads")
OUTPUT_DIR = os.path.join("static", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# トップページ
@app.route("/")
def index():
    return render_template("index.html")

# 画像生成
@app.route("/generate", methods=["POST"])
def generate():
    image = request.files.get("image")
    instruction = request.form.get("instruction", "")

    if not image:
        return "画像がアップロードされていません", 400

    # ファイル名をユニークに
    uid = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_input.png")
    output_path = os.path.join(OUTPUT_DIR, f"{uid}_result.png")

    image.save(input_path)

    # --- GPT連携 ---
    replacements = parse_instruction_with_gpt(instruction)
    print("GPT置換ペア:", replacements)

    # GPTが何も返さなかった場合の保険
    if not replacements:
        replacements = []

    # --- 画像処理 ---
    fill_all_speech_balloons_white(
        image_path=input_path,
        output_path=output_path,
        replacements=replacements
    )

    # 元画像と結果画像を画面に返す
    return render_template(
        "index.html",
        original_image=input_path,
        result_image=output_path,
        instruction=instruction
    )

# 起動
if __name__ == "__main__":
    app.run(debug=True)

def run_mmocr_on_balloons(balloon_images):
    # ★ ここで初めて import ★
    from mmocr.apis import MMOCRInferencer

    ocr = MMOCRInferencer(
        det="dbnetpp",
        rec="sar",
        device="cpu"  # WindowsはCPU安定
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

