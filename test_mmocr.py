from mmocr.apis import MMOCRInferencer
import glob

inferencer = MMOCRInferencer(
    det='dbnetpp',
    rec='sar',
    device='cuda'  # GPU
)

for path in glob.glob("balloons/*.png"):
    result = inferencer(path)
    pred = result["predictions"][0]

    print("🗨", path)
    for text, score in zip(pred["rec_texts"], pred["rec_scores"]):
        if score >= 0.4:
            print("  ", text, f"({score:.2f})")
