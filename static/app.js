const form = document.getElementById("signalForm");
const symbolInput = document.getElementById("symbol");
const holdMinutesInput = document.getElementById("holdMinutes");
const candleIntervalInput = document.getElementById("candleInterval");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");

const actionBadge = document.getElementById("actionBadge");
const confidenceText = document.getElementById("confidenceText");
const winningProbability = document.getElementById("winningProbability");
const holdTimeOut = document.getElementById("holdTimeOut");
const stockSuggestion = document.getElementById("stockSuggestion");
const symbolOut = document.getElementById("symbolOut");
const spotPrice = document.getElementById("spotPrice");
const intervalOut = document.getElementById("intervalOut");
const probUp = document.getElementById("probUp");
const probDown = document.getElementById("probDown");
const optionAction = document.getElementById("optionAction");
const optionHint = document.getElementById("optionHint");
const optionSuggestion = document.getElementById("optionSuggestion");
const modelNote = document.getElementById("modelNote");
const warningBox = document.getElementById("warning");
const decisionFactors = document.getElementById("decisionFactors");
const barUp = document.getElementById("barUp");
const barDown = document.getElementById("barDown");
const snapshotGrid = document.getElementById("snapshotGrid");
const analysisText = document.getElementById("analysisText");
const indicatorLegend = document.getElementById("indicatorLegend");
const liveCallBody = document.getElementById("liveCallBody");
const livePutBody = document.getElementById("livePutBody");
const liveMeta = document.getElementById("liveMeta");
const liveError = document.getElementById("liveError");

function setText(el, value) {
  if (!el) return;
  el.textContent = value;
}

const requiredIds = [
  "signalForm",
  "symbol",
  "holdMinutes",
  "candleInterval",
  "loading",
  "error",
  "result",
  "actionBadge",
  "confidenceText",
  "winningProbability",
  "holdTimeOut",
  "stockSuggestion",
  "symbolOut",
  "spotPrice",
  "intervalOut",
  "probUp",
  "probDown",
  "optionAction",
  "optionHint",
  "optionSuggestion",
  "modelNote",
  "warning",
  "decisionFactors",
  "barUp",
  "barDown",
  "snapshotGrid",
  "analysisText",
  "indicatorLegend",
  "liveCallBody",
  "livePutBody",
  "liveMeta",
  "liveError",
];
const missingIds = requiredIds.filter((id) => !document.getElementById(id));
if (missingIds.length > 0) {
  console.error("Missing DOM IDs:", missingIds.join(", "));
}

function setActionStyle(action) {
  if (!actionBadge) return;
  const map = {
    BUY: { bg: "#d8f5e7", color: "#0b8f57" },
    SELL: { bg: "#fde0dd", color: "#b4231a" },
    HOLD: { bg: "#f7efd8", color: "#6f5a1c" },
  };
  const style = map[action] || { bg: "#ececec", color: "#333" };
  actionBadge.style.backgroundColor = style.bg;
  actionBadge.style.color = style.color;
}

function formatPct(v) {
  return `${(v * 100).toFixed(2)}%`;
}

function valueClass(v) {
  return v >= 0 ? "pos" : "neg";
}

function extractPercent(text) {
  const m = String(text || "").match(/(-?\\d+(?:\\.\\d+)?)%/);
  if (!m) return null;
  return Math.max(0, Math.min(100, Math.abs(Number(m[1]))));
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const symbol = symbolInput.value.trim();
  const holdMinutes = Number(holdMinutesInput.value || 60);
  const candleInterval = (candleIntervalInput.value || "auto").trim().toLowerCase();
  if (!symbol) return;

  loading.classList.remove("hidden");
  errorBox.classList.add("hidden");
  result.classList.add("hidden");

  try {
    const res = await fetch("/api/signal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, hold_minutes: holdMinutes, candle_interval: candleInterval }),
    });

    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "Failed to generate signal");
    }

    setText(actionBadge, data.action);
    setActionStyle(data.action);

    setText(confidenceText, `${(data.confidence * 100).toFixed(2)}%`);
    setText(winningProbability, `${(data.winning_probability * 100).toFixed(2)}%`);
    setText(holdTimeOut, `${data.hold_minutes} min`);
    setText(stockSuggestion, data.stock_suggestion);
    setText(symbolOut, data.symbol);
    setText(spotPrice, `Spot ${data.spot_price}`);
    setText(intervalOut, `Requested ${data.requested_interval} | Used ${data.analysis_interval}`);
    setText(probUp, `UP ${formatPct(data.probability_up)}`);
    setText(probDown, `DOWN ${formatPct(data.probability_down)}`);
    if (barUp) barUp.style.width = `${(data.probability_up * 100).toFixed(2)}%`;
    if (barDown) barDown.style.width = `${(data.probability_down * 100).toFixed(2)}%`;
    setText(optionAction, data.option_action);
    setText(optionHint, data.option_hint);
    setText(optionSuggestion, data.option_suggestion);
    setText(modelNote, data.model_note);
    setText(warningBox, data.warning);
    decisionFactors.innerHTML = "";
    (data.decision_factors || []).forEach((item) => {
      const pct = extractPercent(item);
      const card = document.createElement("div");
      card.className = "factor-card";
      card.innerHTML = `<p class="factor-text">${item}</p>${
        pct !== null
          ? `<div class="mini-track"><div class="mini-fill" style="width:${pct.toFixed(2)}%"></div></div>`
          : ""
      }`;
      decisionFactors.appendChild(card);
    });
    if (data.market_snapshot) {
      const m = data.market_snapshot;
      const orderedKeys = [
        "macd",
        "macd_signal",
        "macd_hist",
        "ema12",
        "ema26",
        "rsi14",
        "bb_width",
        "bb_percent_b",
        "atr14_pct",
      ];
      snapshotGrid.innerHTML = orderedKeys
        .filter((k) => m[k] !== undefined)
        .map(
          (k) =>
            `<div class="metric"><span class="metric-label">${labelByKey(
              k
            )}</span><span class="metric-value ${metricClassByKey(
              k,
              m[k]
            )}">${formatByKey(k, m[k])}</span></div>`
        )
        .join("");
      setText(indicatorLegend, "Trend: EMA | Momentum: MACD, RSI | Volatility: Bollinger, ATR");
    } else {
      snapshotGrid.innerHTML = "";
      setText(indicatorLegend, "");
    }
    setText(analysisText, data.analysis_text || "");

    result.classList.remove("hidden");
  } catch (err) {
    setText(errorBox, err.message);
    errorBox.classList.remove("hidden");
  } finally {
    loading.classList.add("hidden");
  }
});
function formatByKey(key, value) {
  const pctKeys = new Set([
    "ret_1",
    "ret_3",
    "ret_6",
    "price_vs_sma9",
    "price_vs_sma21",
    "vol_chg_5",
    "bb_width",
    "atr14_pct",
  ]);
  if (pctKeys.has(key)) return formatPct(value);
  if (key === "rsi14" || key === "bb_percent_b") return Number(value).toFixed(2);
  if (["ema12", "ema26", "sma_9", "sma_21"].includes(key)) return Number(value).toFixed(2);
  return Number(value).toFixed(4);
}

function labelByKey(key) {
  const labels = {
    macd: "MACD",
    macd_signal: "MACD Signal",
    macd_hist: "MACD Histogram",
    ema12: "EMA 12",
    ema26: "EMA 26",
    rsi14: "RSI (14)",
    bb_width: "Bollinger Width",
    bb_percent_b: "Bollinger %B",
    atr14_pct: "ATR % (14)",
  };
  return labels[key] || key;
}

function metricClassByKey(key, value) {
  if (["ema12", "ema26", "sma_9", "sma_21", "rsi14", "bb_percent_b", "vol_20"].includes(key)) return "";
  return valueClass(value);
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function shortTime(ts) {
  if (!ts) return "-";
  const s = String(ts);
  return s.length >= 16 ? s.slice(0, 16).replace("T", " ") : s;
}

function renderLiveTable(tbody, rows) {
  if (!tbody) return;
  if (!Array.isArray(rows) || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">No rows</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r) => {
      const quoteUrl = esc(r.quote_url || "#");
      const symbol = esc(r.normalized_symbol || r.symbol || "-");
      return `<tr>
        <td>${esc(shortTime(r.scan_time))}</td>
        <td>${symbol}</td>
        <td>${formatPct(Number(r.confidence || 0))}</td>
        <td>${Number(r.spot_price || 0).toFixed(2)}</td>
        <td><a href="${quoteUrl}" target="_blank" rel="noopener noreferrer">Open</a></td>
      </tr>`;
    })
    .join("");
}

async function loadLiveSignals() {
  if (!liveCallBody || !livePutBody) return;
  try {
    if (liveError) liveError.classList.add("hidden");
    const res = await fetch("/api/live-signals?limit=12");
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "Failed to load live signals");
    }
    renderLiveTable(liveCallBody, data.buy_call || []);
    renderLiveTable(livePutBody, data.buy_put || []);
    if (liveMeta) {
      liveMeta.textContent = `Source: ${data.sources.buy_call_csv} and ${data.sources.buy_put_csv} | auto refresh 30s`;
    }
  } catch (err) {
    if (liveError) {
      liveError.textContent = err.message || String(err);
      liveError.classList.remove("hidden");
    }
  }
}

loadLiveSignals();
setInterval(loadLiveSignals, 30000);
