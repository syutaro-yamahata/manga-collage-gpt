from mmocr.apis import MMOCRInferencer

# テキスト検出モデル（吹き出しなどの文字領域を検出）
ocr = MMOCRInferencer('dbnet')

# 対象画像を指定
image_path = "debug_ocr.png"  # ← ここを実際の画像名に置き換えてOK

# 推論を実行（結果画像を保存）
result = ocr(image_path, save_vis=True)

# 結果を出力
print(result['predictions'])
