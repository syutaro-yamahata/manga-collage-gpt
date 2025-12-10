import os
import json
import re
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# 環境変数から読む（.env に設定しておく）
# AZURE_OPENAI_ENDPOINT="https://xxxxxxxx.openai.azure.com/"
# AZURE_OPENAI_API_KEY="xxxxx"
# AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"  ← Azure側で作ったデプロイ名

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
api_version = "2024-12-01-preview"

if not endpoint or not api_key:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY を環境変数に設定してください")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

def _extract_json(text: str):
    """
    モデルが ```json ... ``` で返したり、前後に説明文が付いたときの保険。
    """
    # ```json ... ``` ブロック優先
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    if fence:
        return json.loads(fence.group(1))
    # テキスト全体から最初のJSON配列/オブジェクトを拾う
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if m:
        return json.loads(m.group(1))
    # そのままトライ
    return json.loads(text)

def parse_instruction_with_gpt(user_instruction: str):
    """
    ユーザーの自然文から置換ペアを抽出し、JSONリストを返す。
    例: [{"from":"オレ","to":"私"}, {"from":"勝利","to":"敗北"}]
    """
    system_prompt = (
        "あなたは漫画の吹き出しコラ編集用AIです。"
        "ユーザーが入力した自然文から置換ペアを抽出してください。"
        "出力は必ずJSONリストのみで返してください。"
        "各要素は {\"from\": <文字列>, \"to\": <文字列>} の形にしてください。"
        "余分な説明や文章は出力しないでください。"
    )

    resp = client.chat.completions.create(
        model=deployment,  # Azure: ここはデプロイ名
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
        ],
        temperature=0.3,
        top_p=1.0,
        max_tokens=600,
        # JSONモードを使える環境なら以下を有効化（対応していない場合は外す）
        # response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    try:
        return _extract_json(raw)
    except Exception as e:
        print("GPT出力のJSONパースに失敗しました:", e)
        print("raw:", raw)
        return []
