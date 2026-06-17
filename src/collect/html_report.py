"""
html_report.py - Generate a self-contained HTML report from normalized parquet data.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd
from plotly.offline.offline import get_plotlyjs

from src.collect.normalize import classify_token_type


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in (
        "total_quantity",
        "total_cost",
        "avg_effective_price",
        "avg_payg_price",
        "discount_pct",
        "discount_abs",
        "effective_price_per_1m",
        "quantity",
        "cost",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def _derive_ai_resource_type(df: pd.DataFrame) -> pd.Series:
    if "service_tier" in df.columns:
        return df["service_tier"].fillna("Unknown").astype(str)

    if "meter_name" in df.columns:
        meter = df["meter_name"].fillna("").astype(str)
        meter_l = meter.str.lower()
        labels = []
        for raw, low in zip(meter.tolist(), meter_l.tolist()):
            if "cached" in low or "cache" in low:
                labels.append("Cached Input Tokens")
            elif "prompt" in low or "input" in low:
                labels.append("Input Tokens")
            elif "completion" in low or "output" in low:
                labels.append("Output Tokens")
            elif "ptu" in low:
                labels.append("PTU")
            elif "image" in low or "dall" in low:
                labels.append("Image")
            elif raw:
                labels.append(raw)
            else:
                labels.append("Unknown")
        return pd.Series(labels, index=df.index)

    return pd.Series(["Unknown"] * len(df), index=df.index)


def _records_for_json(df: pd.DataFrame) -> list[dict]:
    qty_col = "total_quantity" if "total_quantity" in df.columns else "quantity"
    cost_col = "total_cost" if "total_cost" in df.columns else "cost"
    eff_col = (
        "effective_price_per_1m"
        if "effective_price_per_1m" in df.columns
        else "avg_effective_price"
    )
    model_col = "product_name" if "product_name" in df.columns else "meter_name"
    meter_col = "meter_name" if "meter_name" in df.columns else "meter"
    sub_col = "subscription_id" if "subscription_id" in df.columns else "subscription"

    rows: list[dict] = []
    for _, row in df.iterrows():
        dt = row.get("date")
        token_type = row.get("token_type")
        if not token_type:
            token_type = classify_token_type(str(row.get(meter_col, "") or ""))
        rows.append(
            {
                "subscription_id": str(row.get(sub_col, "") or ""),
                "meter_name": str(row.get(meter_col, "") or ""),
                "product_name": str(row.get(model_col, "") or ""),
                "ai_resource_type": str(row.get("ai_resource_type", "") or ""),
                "token_type": str(token_type or "Unknown"),
                "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "",
                "total_quantity": float(row.get(qty_col, 0.0) or 0.0),
                "total_cost": float(row.get(cost_col, 0.0) or 0.0),
                "effective_price_per_1m": float(row.get(eff_col, 0.0) or 0.0),
                "avg_effective_price": float(row.get("avg_effective_price", 0.0) or 0.0),
                "avg_payg_price": float(row.get("avg_payg_price", 0.0) or 0.0),
                "discount_pct": float(row.get("discount_pct", 0.0) or 0.0),
                "currency": str(row.get("currency", "") or ""),
            }
        )
    return rows


def generate_html_report(
    input_file: Path,
    output_file: Path,
    title: str = "Azure OpenAI Cost Report",
) -> tuple[int, int]:
    """Generate self-contained HTML report.

    Returns:
        tuple[int, int]: (input_row_count, output_row_count)
    """
    df = pd.read_parquet(input_file)
    input_count = len(df)
    if df.empty:
        raise ValueError("Input parquet file is empty.")

    df = _coerce_columns(df)
    df["ai_resource_type"] = _derive_ai_resource_type(df)

    records = _records_for_json(df)
    subs = sorted({r["subscription_id"] for r in records if r["subscription_id"]})
    resource_types = sorted({r["ai_resource_type"] for r in records if r["ai_resource_type"]})
    token_types = sorted({r["token_type"] for r in records if r["token_type"]})
    currency = next((r["currency"] for r in records if r["currency"]), "")
    title_html = html.escape(title, quote=True)
    sub_options = "".join(
        f'<option value="{html.escape(s, quote=True)}">{html.escape(s, quote=True)}</option>'
        for s in subs
    )
    resource_options = "".join(
        f'<option value="{html.escape(r, quote=True)}">{html.escape(r, quote=True)}</option>'
        for r in resource_types
    )
    token_options = "".join(
        f'<option value="{html.escape(t, quote=True)}">{html.escape(t, quote=True)}</option>'
        for t in token_types
    )
    data_json_b64 = base64.b64encode(
        json.dumps(records, ensure_ascii=True).encode("utf-8")
    ).decode("ascii")
    currency_json = json.dumps(currency, ensure_ascii=True).replace("</", "<\\/")

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src 'none'; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>{title_html}</title>
  <style>
    body {{
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      margin: 20px;
      color: #111827;
      background: #f9fafb;
    }}
    .title {{ margin: 0 0 4px; }}
    .subtitle {{ color: #4b5563; margin: 0 0 16px; }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 12px 0 18px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 14px;
    }}
    .label {{ font-size: 12px; color: #6b7280; margin: 0; }}
    .value {{ font-size: 24px; font-weight: 700; margin: 6px 0 0; }}
    .panel {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f3f4f6; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1 class="title">💰 {title_html}</h1>
  <p class="subtitle">Token usage, effective prices, discounts, and model usage breakdown</p>

  <div class="filters">
    <label>
      Subscription
      <select id="subscription-filter">
        <option value="__all__">All subscriptions</option>
        {sub_options}
      </select>
    </label>
    <label>
      AI resource type
      <select id="resource-filter">
        <option value="__all__">All resource types</option>
        {resource_options}
      </select>
    </label>
    <label>
      Token type
      <select id="token-filter">
        <option value="__all__">All token types</option>
        {token_options}
      </select>
    </label>
  </div>

  <div class="card-grid">
    <div class="card"><p class="label">Total Tokens</p><p class="value" id="kpi-tokens">0</p></div>
    <div class="card"><p class="label">Cached Input Tokens</p><p class="value" id="kpi-cached">0</p></div>
    <div class="card"><p class="label">Input Tokens</p><p class="value" id="kpi-input">0</p></div>
    <div class="card"><p class="label">Output Tokens</p><p class="value" id="kpi-output">0</p></div>
    <div class="card"><p class="label">Total Cost</p><p class="value" id="kpi-cost">0</p></div>
    <div class="card"><p class="label">Avg Eff. Price / 1M</p><p class="value" id="kpi-eff">0</p></div>
    <div class="card"><p class="label">Avg Discount %</p><p class="value" id="kpi-disc">0</p></div>
  </div>

  <div class="panel">
    <h3>Cost by Meter per Subscription</h3>
    <div id="chart-cost-by-meter"></div>
  </div>
  <div class="panel">
    <h3>Daily Cost Trend</h3>
    <div id="chart-daily-cost"></div>
  </div>
  <div class="panel">
    <h3>Top 20 Meters by Cost</h3>
    <div id="table-top20"></div>
  </div>
  <div class="panel">
    <h3>Model Usage Breakdown (Subscription + AI Resource Type)</h3>
    <div id="table-models"></div>
  </div>

  <p class="muted" id="footnote"></p>

  <script>{get_plotlyjs()}</script>
  <script>
    const DATA = JSON.parse(atob("{data_json_b64}"));
    const CURRENCY = {currency_json};
    const subFilter = document.getElementById("subscription-filter");
    const resourceFilter = document.getElementById("resource-filter");
    const tokenFilter = document.getElementById("token-filter");

    const INT_FMT = new Intl.NumberFormat();
    const FMT_2 = new Intl.NumberFormat(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    const FMT_4 = new Intl.NumberFormat(undefined, {{minimumFractionDigits: 4, maximumFractionDigits: 4}});
    const FMT_6 = new Intl.NumberFormat(undefined, {{minimumFractionDigits: 6, maximumFractionDigits: 6}});
    const fmtInt = (v) => INT_FMT.format(Math.round(v || 0));
    const fmt2 = (v) => FMT_2.format(v || 0);
    const fmt4 = (v) => FMT_4.format(v || 0);
    const fmt6 = (v) => FMT_6.format(v || 0);

    function applyFilters(rows) {{
      const s = subFilter.value;
      const r = resourceFilter.value;
      const t = tokenFilter.value;
      return rows.filter(x =>
        (s === "__all__" || x.subscription_id === s) &&
        (r === "__all__" || x.ai_resource_type === r) &&
        (t === "__all__" || x.token_type === t)
      );
    }}

    function groupSum(rows, keys, valueKey) {{
      const m = new Map();
      for (const row of rows) {{
        const k = JSON.stringify(keys.map(c => row[c] || ""));
        m.set(k, (m.get(k) || 0) + (Number(row[valueKey]) || 0));
      }}
      return m;
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function renderTable(containerId, headers, rows) {{
      const container = document.getElementById(containerId);
      if (!rows.length) {{
        container.innerHTML = "<p class='muted'>No data for current filters.</p>";
        return;
      }}
      const thead = "<thead><tr>" + headers.map(h => `<th>${{escapeHtml(h)}}</th>`).join("") + "</tr></thead>";
      const tbody = "<tbody>" + rows.map(r => "<tr>" + r.map(c => `<td>${{escapeHtml(c)}}</td>`).join("") + "</tr>").join("") + "</tbody>";
      container.innerHTML = `<table>${{thead}}${{tbody}}</table>`;
    }}

    function rerender() {{
      const filtered = applyFilters(DATA);
      const tokens = filtered.reduce((a, b) => a + (Number(b.total_quantity) || 0), 0);
      const cost = filtered.reduce((a, b) => a + (Number(b.total_cost) || 0), 0);
      const weightedEff = filtered.reduce(
        (a, b) => a + ((Number(b.effective_price_per_1m) || 0) * (Number(b.total_quantity) || 0)),
        0
      );
      const avgEff = tokens > 0 ? weightedEff / tokens : 0;
      const avgDisc = filtered.length ? filtered.reduce((a, b) => a + (Number(b.discount_pct) || 0), 0) / filtered.length : 0;

      const tokensByType = (type) => filtered.reduce(
        (a, b) => a + (b.token_type === type ? (Number(b.total_quantity) || 0) : 0),
        0
      );

      document.getElementById("kpi-tokens").textContent = fmtInt(tokens);
      document.getElementById("kpi-cached").textContent = fmtInt(tokensByType("Cached Input Tokens"));
      document.getElementById("kpi-input").textContent = fmtInt(tokensByType("Input Tokens"));
      document.getElementById("kpi-output").textContent = fmtInt(tokensByType("Output Tokens"));
      document.getElementById("kpi-cost").textContent = `${{fmt2(cost)}} ${{CURRENCY}}`;
      document.getElementById("kpi-eff").textContent = `${{fmt4(avgEff)}} ${{CURRENCY}}`;
      document.getElementById("kpi-disc").textContent = `${{fmt2(avgDisc)}}%`;
      document.getElementById("footnote").textContent = `Rows shown: ${{filtered.length.toLocaleString()}} / ${{DATA.length.toLocaleString()}}`;

      const meterKeys = groupSum(filtered, ["subscription_id", "meter_name"], "total_cost");
      const subscriptions = [...new Set(filtered.map(x => x.subscription_id || "unknown"))];
      const meters = [...new Set(filtered.map(x => x.meter_name || "unknown"))];
      const traces = meters.map(meter => {{
        const y = subscriptions.map(sub => meterKeys.get(JSON.stringify([sub, meter])) || 0);
        return {{
          type: "bar",
          name: meter,
          x: subscriptions,
          y
        }};
      }});
      Plotly.newPlot("chart-cost-by-meter", traces, {{
        barmode: "stack",
        margin: {{t: 10}},
        yaxis: {{title: `Cost (${{CURRENCY}})`}},
        xaxis: {{title: "Subscription"}}
      }}, {{responsive: true}});

      const dailyMap = groupSum(filtered, ["date"], "total_cost");
      const days = [...dailyMap.keys()].filter(Boolean).sort();
      const dailyY = days.map(d => dailyMap.get(d) || 0);
      Plotly.newPlot("chart-daily-cost", [{{
        type: "scatter",
        mode: "lines+markers",
        x: days,
        y: dailyY,
        name: "Daily Cost"
      }}], {{
        margin: {{t: 10}},
        yaxis: {{title: `Cost (${{CURRENCY}})`}},
        xaxis: {{title: "Day"}}
      }}, {{responsive: true}});

      const top20 = [...filtered]
        .sort((a, b) => (Number(b.total_cost) || 0) - (Number(a.total_cost) || 0))
        .slice(0, 20)
        .map(r => [
          r.meter_name || "",
          r.subscription_id || "",
          r.product_name || "",
          fmtInt(r.total_quantity),
          fmt2(r.total_cost),
          fmt6(r.avg_effective_price),
          fmt6(r.avg_payg_price),
          `${{fmt2(r.discount_pct)}}%`,
          fmt6(r.effective_price_per_1m),
          r.currency || ""
        ]);
      renderTable(
        "table-top20",
        ["Meter", "Subscription", "Model", "Total Quantity", "Total Cost", "Avg Effective Price", "Avg PAYG Price", "Discount %", "Eff. Price / 1M", "Currency"],
        top20
      );

      const modelMap = new Map();
      for (const r of filtered) {{
        const k = `${{r.subscription_id}}||${{r.ai_resource_type}}||${{r.product_name}}`;
        if (!modelMap.has(k)) modelMap.set(k, {{tokens: 0, cost: 0}});
        const cur = modelMap.get(k);
        cur.tokens += Number(r.total_quantity) || 0;
        cur.cost += Number(r.total_cost) || 0;
      }}
      const modelRows = [...modelMap.entries()]
        .map(([k, v]) => {{
          const [sub, rt, model] = k.split("||");
          return {{
            cost: Number(v.cost) || 0,
            cells: [sub, rt, model, fmtInt(v.tokens), fmt2(v.cost), CURRENCY]
          }};
        }})
        .sort((a, b) => b.cost - a.cost)
        .map(x => x.cells);
      renderTable(
        "table-models",
        ["Subscription", "AI Resource Type", "Model", "Total Tokens", "Total Cost", "Currency"],
        modelRows
      );
    }}

    subFilter.addEventListener("change", rerender);
    resourceFilter.addEventListener("change", rerender);
    tokenFilter.addEventListener("change", rerender);
    rerender();
  </script>
</body>
</html>
"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding="utf-8")
    return input_count, len(df)
