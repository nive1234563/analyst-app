# causal_ai.py
import os, json
from typing import Any, Dict, Optional
from openai import OpenAI

def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    # 
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

def generate_causal_ai_insights(
    facts: Dict[str, Any],
    custom_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 700,
    temperature: float = 0.3,
) -> str:
    client = _client()
    base = (
        "You are a senior data scientist. Using the causal facts, produce:\n"
        "• 5–8 concise, business-facing insights (what to do, for whom, thresholds)\n"
        "• 3–5 next experiments/targeting rules\n"
        "• 1–2 guardrails (overlap, leakage, stability)\n"
        "Keep it crisp and actionable and within 300 words. Avoid jargon."
    )
    if custom_prompt:
        base += "\n\nBusiness context: " + custom_prompt

    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write crisp, actionable analytics summaries for business users."},
            {"role": "user", "content": base + "\n\n---\nCAUSAL FACTS JSON:\n" + facts_json}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
