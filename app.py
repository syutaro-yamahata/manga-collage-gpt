# Flask本体と各種ユーティリティ関数をインポート
from flask import Flask, render_template, request, send_file
import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "FuriganaDetection/lib")))

from dotenv import load_dotenv
from collage import fill_all_speech_balloons_white   # 自作の画像処理モジュールをインポート
from gpt_helper import parse_instruction_with_gpt  # GPT連携はコメントアウト中

# .envファイルの環境変数を読み込む（APIキーなど）
load_dotenv()

# Flaskアプリケーションのインスタンス作成
app = Flask(__name__)
# アップロードされた画像を保存するディレクトリの指定
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ルートURLにアクセスされたときにindex.htmlを表示
@app.route('/')
def index():
    return render_template('index.html')

# フォーム送信時（POST）に呼び出されるエンドポイント
@app.route('/generate', methods=['POST'])
def generate():
    # フォームから送られた画像ファイルとテキスト指示を取得
    image = request.files['image']
    instruction = request.form['instruction']
    
    # 画像を指定フォルダに保存
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # GPTを使わず、手動で置換ペアを定義（「オレ」→「私」）
    #replacements = [
       # {"from": "オレ", "to": "私"},
       # {"from": "勝利", "to": "敗北"}
    # ]
    #print("固定置換ペア:", replacements)

    # GPTを使いたい場合は有効にする ↓
    
    replacements = parse_instruction_with_gpt(instruction)
    print("GPTが抽出した置換ペア:", replacements)

    # 画像処理（OCR → テキスト置換）を実行
    result_path = process_image(image_path, replacements)

    # 生成された画像を返却（ブラウザに表示）
    return send_file(result_path, mimetype='image/png')

# Flaskアプリを開発モードで起動
if __name__ == '__main__':
    app.run(debug=True)
