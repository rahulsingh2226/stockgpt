# app.py — StockGPT (prices, charts, options) with intent routing
import os, re, io
from textwrap import dedent
import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import requests

# =============== Page ===============
st.set_page_config(page_title="StockGPT", page_icon="📈", layout="wide")
st.title("📈 StockGPT — Finance Chat + Quotes/Charts/Options (Local)")

st.caption(
    "Ask naturally (e.g., *price of AAPL*, *show NVDA chart 6 months*, *TSLA option chain*, "
    "*SPY greeks*, *AAPL IV smile*).  \n"
    "Slash commands also work: `/price TICKER`, `/chart TICKER [period] [interval]`, "
    "`/chain TICKER [YYYY-MM-DD]`, `/greeks TICKER [YYYY-MM-DD]`, `/smile TICKER [YYYY-MM-DD]`, `/heat TICKER [YYYY-MM-DD]`, `/help`."
)

# =============== Config ===============
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "stockgpt")

PERIODS   = {"1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"}
INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}

SYSTEM_PROMPT = dedent("""
You are StockGPT, a finance-only assistant for stocks, ETFs, indices, options, crypto, markets,
technicals and fundamentals. Do not answer non-finance topics. If a user asks something outside finance,
reply exactly: "Sorry, I can’t help you with this."
Be concise and educational. Do not provide personal investment advice.
""").strip()

FINANCE_RGX = re.compile(
    r"(stock|equit|option|put|call|greek|delta|gamma|theta|vega|rho|"
    r"iv|volatil|implied|chain|strike|expiry|quote|price|chart|candle|"
    r"moving average|rsi|macd|fundamental|earnings|dividend|crypto|bitcoin|ethereum|"
    r"ticker|market|index|etf|open interest|volume|heatmap)", re.I
)

TICKER_RGX = re.compile(r"\b([A-Z]{1,6}(?:[-.][A-Z]{1,5})?|[A-Z]{2,5}-USD|\^[A-Z]{1,5})\b")

# =============== Helpers ===============
def finance_only(text: str) -> bool:
    return bool(FINANCE_RGX.search(text or "")) or text.strip().startswith("/")

def note_delay():
    st.caption("Data via Yahoo Finance (free). US equities typically delayed ~15–20 minutes; crypto is closer to real-time.")

def extract_ticker(text: str) -> str | None:
    # choose the first candidate that returns data
    cands = [m.group(1).upper() for m in TICKER_RGX.finditer(text or "")]
    for tk in cands[:3]:
        try:
            df = yf.download(tk, period="5d", progress=False)
            if not df.empty:
                return tk
        except Exception:
            pass
    return cands[0].upper() if cands else None

def parse_period_interval(text: str):
    text = (text or "").lower()
    # common phrases -> period / interval
    if "daily" in text or "1 day" in text:   return "1mo", "1d"
    if "weekly" in text or "1 week" in text: return "6mo", "1wk"
    if "monthly" in text:                    return "2y",  "1mo"
    if "ytd" in text:                        return "ytd", "1d"
    if "6 month" in text or "6mo" in text:   return "6mo", "1d"
    if "3 month" in text or "3mo" in text:   return "3mo", "1d"
    if "1 year" in text or "1y" in text:     return "1y",  "1d"
    if "max" in text:                        return "max", "1wk"
    # default
    return "1mo", "1d"

@st.cache_data(show_spinner=False, ttl=60)
def quote_latest(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    h1d = tk.history(period="1d", prepost=True)
    last = float(h1d["Close"].iloc[-1]) if not h1d.empty else None
    fi = getattr(tk, "fast_info", {}) if hasattr(tk, "fast_info") else {}
    cur = fi.get("currency") if isinstance(fi, dict) else getattr(fi, "currency", None)
    return {"symbol": ticker.upper(), "last": last, "currency": cur}

@st.cache_data(show_spinner=False, ttl=60)
def price_df(ticker: str, period="1mo", interval="1d") -> pd.DataFrame:
    return yf.download(ticker, auto_adjust=True, progress=False, period=period, interval=interval)

def plot_price(df: pd.DataFrame, ticker: str, period: str, interval: str):
    if df is None or df.empty:
        st.warning("No price data returned.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], lw=1.5)
    ax.set_title(f"{ticker.upper()} — {period} / {interval}")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

@st.cache_data(show_spinner=False, ttl=120)
def expiries(ticker: str) -> list[str]:
    try:
        return yf.Ticker(ticker).options or []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=120)
def option_chain(ticker: str, expiry: str | None):
    tk = yf.Ticker(ticker)
    exps = tk.options or []
    if not exps:
        return pd.DataFrame(), pd.DataFrame(), ""
    if not expiry or expiry not in exps:
        expiry = exps[0]
    oc = tk.option_chain(expiry)
    calls, puts = oc.calls.copy(), oc.puts.copy()
    for df in (calls, puts):
        for col in ["impliedVolatility","delta","gamma","theta","vega","rho","bid","ask","lastPrice","volume","openInterest","strike"]:
            if col not in df.columns:
                df[col] = np.nan
        df["mid"] = np.where(pd.notna(df["bid"]) & pd.notna(df["ask"]), (df["bid"]+df["ask"])/2.0, df["lastPrice"])
    return calls, puts, expiry

def df_download_button(df: pd.DataFrame, label: str, filename: str):
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

def nearest_atm_slice(calls, puts, spot, width_pct=0.10, max_rows=40):
    if spot and not np.isnan(spot):
        lo, hi = spot*(1-width_pct), spot*(1+width_pct)
        c = calls[(calls["strike"]>=lo)&(calls["strike"]<=hi)].copy()
        p = puts[(puts["strike"]>=lo)&(puts["strike"]<=hi)].copy()
    else:
        c = calls.nlargest(max_rows//2, "volume").copy()
        p = puts.nlargest(max_rows//2, "volume").copy()
    c["type"]="CALL"; p["type"]="PUT"
    return pd.concat([c,p], ignore_index=True).head(max_rows)

def show_greeks_table(df, expiry, ticker):
    if df.empty:
        st.warning("No options in range.")
        return
    cols = ["type","strike","mid","impliedVolatility","delta","gamma","theta","vega","rho","volume","openInterest"]
    view = df[cols].rename(columns={"impliedVolatility":"IV","openInterest":"OI","volume":"Vol"}).copy()
    if "IV" in view.columns: view["IV"] = (100*view["IV"]).round(2)
    for c in ("delta","gamma","theta","vega","rho"):
        if c in view.columns: view[c] = view[c].round(4)
    if "mid" in view.columns: view["mid"] = view["mid"].round(2)
    st.subheader(f"Greeks — {ticker} ({expiry})")
    st.dataframe(view.sort_values(["type","strike"]).reset_index(drop=True), use_container_width=True)
    df_download_button(view, "⬇️ Download Greeks (CSV)", f"{ticker}_{expiry}_greeks.csv")

def plot_iv_smile(calls, puts, expiry, spot, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    if not calls.empty: ax.scatter(calls["strike"], 100*calls["impliedVolatility"], s=18, label="Calls IV")
    if not puts.empty:  ax.scatter(puts["strike"], 100*puts["impliedVolatility"], s=18, label="Puts IV")
    if spot and not np.isnan(spot):
        ax.axvline(float(spot), color="gray", lw=1, ls="--", alpha=0.6, label="Spot")
    ax.set_title(f"{ticker} — IV Smile ({expiry})"); ax.set_xlabel("Strike"); ax.set_ylabel("IV (%)")
    ax.grid(True, alpha=0.25); ax.legend()
    st.pyplot(fig)

def plot_oi_vol_heatmaps(calls, puts, ticker, expiry):
    if calls.empty and puts.empty:
        st.warning("No options to chart.")
        return
    strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
    tab = pd.DataFrame(index=strikes)
    def add(name, df, col):
        if not df.empty and col in df.columns:
            tab[name] = df.set_index("strike")[col]
    add("CALL_OI", calls, "openInterest"); add("PUT_OI", puts, "openInterest")
    add("CALL_Vol",calls,"volume");       add("PUT_Vol", puts,"volume")
    tab = tab.fillna(0)
    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(np.vstack([norm(tab["CALL_OI"]).values, norm(tab["PUT_OI"]).values]), aspect="auto")
    axes[0].set_title(f"{ticker} {expiry} — Open Interest (CALL/PUT)"); axes[0].set_yticks([0,1]); axes[0].set_yticklabels(["CALL","PUT"])
    axes[1].imshow(np.vstack([norm(tab["CALL_Vol"]).values, norm(tab["PUT_Vol"]).values]), aspect="auto")
    axes[1].set_title(f"{ticker} {expiry} — Volume (CALL/PUT)"); axes[1].set_yticks([0,1]); axes[1].set_yticklabels(["CALL","PUT"])
    for ax in axes: ax.set_xlabel("Strike (index)")
    fig.tight_layout(); st.pyplot(fig)

def ollama_chat(messages: list[dict]) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()

# =============== Chat state ===============
if "history" not in st.session_state:
    st.session_state.history = [{"role":"system","content":SYSTEM_PROMPT}]

q = st.chat_input("Ask about stocks, options, crypto… (try: price of AAPL)")
if q:
    st.chat_message("user").markdown(q)
    st.session_state.history.append({"role":"user","content":q})

for m in st.session_state.history[1:]:
    if m["role"] == "assistant":
        st.chat_message("assistant").markdown(m["content"])

def respond(text: str):
    st.chat_message("assistant").markdown(text)
    st.session_state.history.append({"role":"assistant","content":text})

# =============== Intent routing ===============
def handle_price(tk: str):
    info = quote_latest(tk)
    if info["last"] is None:
        respond(f"Couldn’t fetch a price for **{tk}**.")
    else:
        respond(f"**{info['symbol']}** — Last: **{info['last']}** {info.get('currency') or ''}")
        note_delay()

def handle_chart(tk: str, period: str, interval: str):
    with st.chat_message("assistant"):
        st.markdown(f"**{tk}** price chart — *{period}* / *{interval}*")
        df = price_df(tk, period=period, interval=interval)
        plot_price(df, tk, period, interval)
        note_delay()
    st.session_state.history.append({"role":"assistant","content":f"Chart shown for {tk} ({period}/{interval})."})

def handle_chain_like(kind: str, tk: str, exp: str | None):
    info = quote_latest(tk); spot = info["last"]
    calls, puts, expiry = option_chain(tk, exp)
    if expiry == "":
        respond(f"No listed options for **{tk}**.")
        return
    if kind == "chain":
        df = (pd.concat([calls.assign(type="CALL"), puts.assign(type="PUT")], ignore_index=True)
                .loc[:,["type","strike","mid","impliedVolatility","volume","openInterest"]]
                .rename(columns={"impliedVolatility":"IV","openInterest":"OI","volume":"Vol"}))
        df["IV"] = (100*df["IV"]).round(2); df["mid"]=df["mid"].round(2)
        with st.chat_message("assistant"):
            st.markdown(f"**{tk}** option chain — **{expiry}**")
            st.dataframe(df.sort_values(["type","strike"]).reset_index(drop=True), use_container_width=True)
            df_download_button(df, "⬇️ Download chain (CSV)", f"{tk}_{expiry}_chain.csv")
            note_delay()
        st.session_state.history.append({"role":"assistant","content":f"Chain shown for {tk} ({expiry})."})
    elif kind == "greeks":
        near = nearest_atm_slice(calls, puts, spot, width_pct=0.10, max_rows=40)
        with st.chat_message("assistant"):
            show_greeks_table(near, expiry, tk)
            note_delay()
        st.session_state.history.append({"role":"assistant","content":f"Greeks shown for {tk} ({expiry})."})
    elif kind == "smile":
        with st.chat_message("assistant"):
            plot_iv_smile(calls, puts, expiry, spot, tk); note_delay()
        st.session_state.history.append({"role":"assistant","content":f"IV smile for {tk} ({expiry})."})
    elif kind == "heat":
        with st.chat_message("assistant"):
            plot_oi_vol_heatmaps(calls, puts, tk, expiry); note_delay()
        st.session_state.history.append({"role":"assistant","content":f"OI/Vol heatmaps for {tk} ({expiry})."})

def dispatch(text: str):
    low = text.lower().strip()

    # Slash commands first
    if low.startswith("/help"):
        respond(dedent("""
        **Commands**
        - `/price TICKER`
        - `/chart TICKER [period] [interval]`  e.g. `/chart NVDA 6mo 1d`
        - `/chain TICKER [YYYY-MM-DD]`
        - `/greeks TICKER [YYYY-MM-DD]`
        - `/smile TICKER [YYYY-MM-DD]`
        - `/heat TICKER [YYYY-MM-DD]`
        """).strip())
        return

    m = re.match(r"^/price\s+([A-Za-z.^-]{1,15})$", low, re.I)
    if m:
        return handle_price(m.group(1).upper())

    m = re.match(r"^/chart\s+([A-Za-z.^-]{1,15})(?:\s+(\w+))?(?:\s+(\w+))?$", low, re.I)
    if m:
        tk = m.group(1).upper()
        period = (m.group(2) or "1mo").lower()
        interval = (m.group(3) or "1d").lower()
        if period not in PERIODS: period = "1mo"
        if interval not in INTERVALS: interval = "1d"
        return handle_chart(tk, period, interval)

    m = re.match(r"^/(chain|greeks|smile|heat)\s+([A-Za-z.^-]{1,15})(?:\s+(\d{4}-\d{2}-\d{2}))?$", low, re.I)
    if m:
        return handle_chain_like(m.group(1).lower(), m.group(2).upper(), m.group(3))

    # Natural language intents (no slashes)
    if any(w in low for w in ["price","quote","last","how is","what's","whats"]) and extract_ticker(text):
        tk = extract_ticker(text)
        return handle_price(tk)

    if "chart" in low or "graph" in low or "plot" in low:
        tk = extract_ticker(text)
        if tk:
            p,i = parse_period_interval(text)
            return handle_chart(tk, p, i)

    if any(w in low for w in ["option chain","options chain","chain","strikes","expiries"]):
        tk = extract_ticker(text)
        if tk:
            exp = None
            m2 = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", low)
            if m2: exp = m2.group(1)
            return handle_chain_like("chain", tk, exp)

    if "greeks" in low or any(w in low for w in ["delta","gamma","theta","vega","rho","iv"]):
        tk = extract_ticker(text)
        if tk:
            m2 = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", low)
            return handle_chain_like("greeks", tk, m2.group(1) if m2 else None)

    if "smile" in low and extract_ticker(text):
        tk = extract_ticker(text)
        m2 = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", low)
        return handle_chain_like("smile", tk, m2.group(1) if m2 else None)

    if "heatmap" in low or ("open interest" in low and "heat" in low):
        tk = extract_ticker(text)
        if tk:
            m2 = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", low)
            return handle_chain_like("heat", tk, m2.group(1) if m2 else None)

    # Guardrail / LLM fallback
    if not finance_only(text):
        respond("Sorry, I can’t help you with this.")
        return

    # Finance question -> Ollama
    try:
        reply = ollama_chat(st.session_state.history)
    except Exception as e:
        reply = f"(local model error) {e}"
    respond(reply)

# Run dispatcher on newest user turn
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    dispatch(st.session_state.history[-1]["content"])
