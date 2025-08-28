# insight_generator.py
import openai
import os
import json
from openai import OpenAI

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

openai.api_key = os.getenv("")

def generate_ai_insights(profiling_output, custom_prompt=None):
    """
    Send ydata_profiling results to GPT and fetch deep insights.
    profiling_output: dict or str (output from ydata_profiling.to_json())
    custom_prompt: optional user instruction
    """
    if isinstance(profiling_output, dict):
        profiling_output = json.dumps(profiling_output, indent=2)

    base_prompt = """
    You are a data analyst. Analyze the profiling report and provide deep insights
    related to business alone in 200 words .
    Go beyond summary — Provide business-relevant interpretations.
    """

    if custom_prompt:
        base_prompt += "\n\nUser context: " + custom_prompt

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # faster, cheaper model — switch to gpt-4 if needed
        messages=[
            {"role": "system", "content": "You are a data analyst helping summarize data for dashboards."},
            {"role": "user", "content": base_prompt + "\n\nHere is the profiling report:\n" + profiling_output}
        ],
        max_tokens=800,  # larger to allow deep insights
        temperature=0.5
    )

    return response.choices[0].message.content

