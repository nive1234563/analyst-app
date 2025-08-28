// src/api/causal.js
const API = "http://localhost:8000";

export async function trainCausal(payload) {
  const res = await fetch(`${API}/causal/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function upliftBySegment(modelId, segmentCol = "region", kQuantiles = 10) {
  const res = await fetch(`${API}/causal/uplift_by_segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, segment_col: segmentCol, k_quantiles: kQuantiles }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function counterfactualCards(modelId, k = 5, policy = "top_k", threshold = null) {
  const res = await fetch(`${API}/causal/counterfactual_cards`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, k, policy, threshold }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
