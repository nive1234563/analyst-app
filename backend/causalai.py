# causal_ai.py
import os
import json
from typing import Any, Dict, Optional
from openai import OpenAI

def _openai_client() -> OpenAI:
    # Expect OPENAI_API_KEY in env; never hardcode keys
    
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    return OpenAI(api_key=api_key)

def generate_causal_ai_insights(
    facts: Dict[str, Any],
    custom_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """
    Turn causal results into crisp, business-facing insights.

    'facts' should include keys like:
      - model_id, estimator, summary, effects (ATE/ATT/ATC)
      - segment_uplift, deciles, top_counterfactual_cards
    """
    client = _openai_client()

    base = (
        "You are a senior data scientist specializing in causal inference and uplift modeling. "
        "Given the causal analysis facts, produce:\n"
        "1) 5–8 bullet insights tying treatment effects to business actions (be specific, avoid jargon)\n"
        "2) A short playbook (3–5 bullets) for next experiments or targeting (thresholds/segments)\n"
        "3) 1–2 guardrails about assumptions/overlap/leakage\n"
        "Keep it concise and actionable and keep it within 300 words."
    )
    if custom_prompt:
        base += "\n\nUser context: " + custom_prompt

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
