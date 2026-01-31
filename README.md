# 🗨️ Manga Collage GPT
### Phase 1: High-Precision Speech Balloon Eraser

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Framework-000000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-FF385C.svg?style=for-the-badge&logo=ultralytics&logoColor=white)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **漫画の「吹き出し」をAIでハックする。** > 枠線や背景を一切傷つけず、中のセリフだけを魔法のように消去。次世代の漫画編集・翻訳パイプラインの第一歩がここに。

---

## 🎨 現在の到達点: "Perfect Blanking"
現在は **Phase 1: Speech Balloon Eraser** です。  
単なるベタ塗りではなく、AIが「マンガの構造」を理解し、枠線を守りながら中身をクリアにする高品質な画像処理パイプラインを構築しました。

---

## 🔄 技術フロー (Technical Workflow)

本プロジェクトは、以下の多段的なプロセスを経て「高品質な白抜き」を実現しています。

```mermaid
graph TD
    %% ユーザー入力
    User((ユーザー)) -- "1. 元画像アップロード" --> WebApp[Flask Web Interface]

    subgraph "AI解析フェーズ (Analysis)"
        WebApp --> YOLO[YOLOv8: 吹き出し検出]
        YOLO -- "2. 座標取得" --> Analysis
        Analysis["・吹き出しのバウンディングボックス<br>・枠線の位置特定"]
    end

    subgraph "画像処理フェーズ (Processing)"
        Analysis --> Logic{枠線保護ロジック}
        Logic -- "3. 外周10%をマスク除外" --> Cleaning[OpenCV: 文字消去]
        Cleaning --> Fill[高精度白抜き処理]
    end

    subgraph "外部連携準備 (Extensions)"
        Fill --> VisionAPI[Google Cloud Vision: OCR]
        VisionAPI --> GPT[Azure OpenAI: 文脈理解/翻訳]
    end

    %% 出力
    Fill -- "4. 編集済み画像 (Blanked)" --> User
    
    style User fill:#ffdf00,stroke:#000,stroke-width:2px
    style WebApp fill:#fff,stroke:#000,stroke-width:2px
    style YOLO fill:#ff3131,color:#fff,stroke:#000,stroke-width:2px
    style Cleaning fill:#ff3131,color:#fff,stroke:#000,stroke-width:2px
    style Logic fill:#f0f0f0,stroke:#000,stroke-width:2px
