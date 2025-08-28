// src/api/forecast.js
const BASE = "http://localhost:8000";

export async function forecastUploadCsv(file, datasetId = "default") {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("dataset_id", datasetId);
  const res = await fetch(`${BASE}/forecast/upload`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function forecastTrain({ datasetId="default", dateCol="ds", targetCol="y", horizon=30, folds=3 }) {
  const res = await fetch(`${BASE}/forecast/train`, {
    method: "POST",
    headers: { "Content-Type":"application/json" },
    body: JSON.stringify({ dataset_id: datasetId, date_col: dateCol, target_col: targetCol, horizon, folds })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function forecastPredict(datasetId="default", horizon) {
  const url = new URL(`${BASE}/forecast/predict`);
  url.searchParams.set("dataset_id", datasetId);
  if (horizon) url.searchParams.set("horizon", String(horizon));
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function forecastPlot(datasetId="default", kind="forecast") {
  const url = new URL(`${BASE}/forecast/plot/${kind}`);
  url.searchParams.set("dataset_id", datasetId);
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(await res.text());
  const { image } = await res.json();
  return `data:image/png;base64,${image}`;
}

export async function forecastInsights(datasetId="default") {
  const res = await fetch(`${BASE}/forecast/insights`, {
    method: "POST",
    headers: { "Content-Type":"application/json" },
    body: JSON.stringify({ dataset_id: datasetId })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
