# ==============================
# 漫画ページの文字検出スクリプト
# ==============================

# comics_text_plus から文字検出・認識クラスをインポート
from comics_text_plus.text_extractor import TextExtractor

# 検出器（TextExtractor）のインスタンスを作成（MMOCRモデルを自動ロード）
extractor = TextExtractor()  # 初回だけモデルの自動ダウンロードあり（少し時間がかかる）

# 入力画像のパスを指定（自分の画像に変更OK）
image_path = "path/to/your/image.png"

# 吹き出し・文字・効果音などを検出してOCR（文字認識）を実行
# readtext() は画像中の文字領域と文字内容をまとめて返す
result = extractor.ocr.readtext(image_path)

# 結果を画像ファイルとして保存（※必要なら後で描画処理追加可能）
# 今はコンソールに検出結果を表示する
print("✅ 検出結果:")
for i, item in enumerate(result):
    print(f"{i+1}. {item}")

# メモリを開放（PyTorchがGPU/CPUを使うため）
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n🎉 検出が完了しました！")
