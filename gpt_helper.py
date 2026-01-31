import os
import json
import re
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")


def parse_instruction_with_gpt(user_instruction: str):
    """
    自然文 → 置換ペア [{from,to}, ...]
    """
    system_prompt = (
        "あなたは漫画編集AIです。"
        "ユーザーの指示（例：『オレを私に変えて』）から、置換前と置換後のペアを抽出してください。"
        "出力は必ず以下のJSON配列形式のみで行ってください。余計な解説は不要です。"
        '[{"from": "置換前", "to": "置換後"}]' # ← 形式を具体的に指定
    )
    

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
        ],
        temperature=0.3,
        max_tokens=300,
    )

    text = resp.choices[0].message.content

    # JSONだけ抜く保険
    m = re.search(r"(\[.*\])", text, re.S)
    return json.loads(m.group(1)) if m else []
