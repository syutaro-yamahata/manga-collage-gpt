Manga Collage GPT (Phase 1: Speech Balloon Eraser)
漫画の吹き出しをAIで自動検出し、枠線を保持したまま中のセリフをきれいに消去するツールです。 現在は、高品質な「白抜き」を実現するフェーズにあります。

 現在の到達点
画像内の吹き出しを自動で特定し、「元の絵（枠線）を壊さずに」 中の文字だけを白く塗りつぶす画像処理パイプラインを構築しました。

 主な機能と技術的特徴
AI吹き出し検出: YOLOv8 モデルを使用して、様々な形状の吹き出しを瞬時に特定します。

枠線保護ロジック: YOLOの検出範囲の端10%を保護領域とし、吹き出しの輪郭が消失するのを防いでいます。

高精度な文字消去: OpenCVを活用し、枠内の「黒い文字（ドット）」のみを狙って白に変換。ベタ塗り感のない自然な仕上がりを実現しました。

Google Cloud Vision 連携: 裏側ではOCR（文字認識）の準備も整っており、セリフの自動取得が可能です。

🛠 技術スタック
Language: Python 3.x

Framework: Flask (Web Interface)

AI/ML: YOLOv8 (ultralytics), Azure OpenAI

Image Processing: OpenCV, Pillow (PIL)

Cloud API: Google Cloud Vision API

 セットアップ・起動方法
1. 依存ライブラリのインストール
Bash

pip install flask pillow opencv-python ultralytics google-cloud-vision openai python-dotenv
2. 認証情報とモデルの配置
credentials.json: Google Cloud サービスアカウントキー

.env: Azure OpenAI のエンドポイントとキーを記述

models/best.pt: 学習済みの YOLO モデル

3. 実行
Bash

python app.py
http://127.0.0.1:5000 にアクセスして画像をアップロードしてください。

📈 今後のロードマップ
[x] YOLOによる吹き出しの自動検出

[x] 枠線を保持した文字消去 (現在地)

[ ] 縦書き/横書きを考慮した自動再描画

[ ] GPTによるセリフの翻訳・トーン変換