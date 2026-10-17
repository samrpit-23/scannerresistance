"""
Nifty 200 Support-Rebound Screener
Run:  pip install streamlit yfinance pandas numpy requests
      streamlit run support_screener.py
"""
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

NSE_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
FALLBACK = ("RELIANCE TCS HDFCBANK ICICIBANK INFY SBIN BHARTIARTL ITC LT AXISBANK KOTAKBANK HINDUNILVR "
            "MARUTI SUNPHARMA TATAMOTORS TATASTEEL NTPC POWERGRID ONGC COALINDIA BAJFINANCE ASIANPAINT "
            "TITAN ULTRACEMCO M&M WIPRO HCLTECH TECHM JSWSTEEL HINDALCO ADANIPORTS CIPLA DRREDDY "
            "BPCL IOC GAIL BANKBARODA PNB CANBK DLF TATAPOWER VEDL SAIL NMDC").split()


@st.cache_data(ttl=86400, show_spinner=False)
def load_universe():
    try:  # 1) official NSE list
        r = requests.get(NSE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
    except Exception:
        pass
    try:  # 2) local nifty200.csv with a 'Symbol' column
        return pd.read_csv("nifty200.csv")["Symbol"].str.strip().tolist()
    except Exception:
        return FALLBACK  # 3) small fallback list


def pivots(s, w, kind):
    r = s.rolling(2 * w + 1, center=True)
    m = (r.min() if kind == "low" else r.max()).values
    v = s.values
    pos = np.where(v == m)[0]
    return pos, v[pos]


def cluster(pos, px, tol):
    groups = []
    for i in np.argsort(px):
        for g in groups:
            if abs(px[i] / np.mean(g[0]) - 1) <= tol:
                g[0].append(px[i]); g[1].append(pos[i]); break
        else:
            groups.append([[px[i]], [pos[i]]])
    return [(np.mean(g[0]), len(g[0])) for g in groups]


def analyze(d, a):
    d = d.dropna()
    if len(d) < 120:
        return None
    O, H, L, C, V = (d[k].values for k in ["Open", "High", "Low", "Close", "Volume"])
    close = C[-1]

    # ---- 1. support zones: clustered swing lows (>=2 touches) ----
    Ls, Hs = L[-200:], H[-200:]
    lp, lv = pivots(pd.Series(Ls), 5, "low")
    cands = []
    for lvl, touches in cluster(lp, lv, 0.015):
        dist = (close / lvl - 1) * 100
        if touches >= 2 and -1 <= dist <= a["dist"]:
            cands.append((touches, -dist, lvl))
    if not cands:
        return None
    touches, _, support = max(cands)

    # ---- 2. nearest resistance above price -> upside room ----
    hp, hv = pivots(pd.Series(H[-250:]), 5, "high")
    above = hv[hv > close * 1.01]
    top = H[-250:].max()
    res = above.min() if len(above) else (top if top > close * 1.01 else close * 1.2)
    upside = (res / close - 1) * 100
    stop = support * 0.985
    risk = max((close / stop - 1) * 100, 0.1)
    rr = upside / risk
    if upside < a["upside"] or rr < a["rr"]:
        return None

    # ---- 3. bullish patterns at support ----
    pats = []
    if H[-1] < H[-2] and L[-1] > L[-2]:
        pats.append("Inside Bar")
    if C[-2] < O[-2] and C[-1] > O[-1] and C[-1] >= O[-2] and O[-1] <= C[-2]:
        pats.append("Bullish Engulfing")
    body, rng = abs(C[-1] - O[-1]), H[-1] - L[-1]
    lower, upper = min(O[-1], C[-1]) - L[-1], H[-1] - max(O[-1], C[-1])
    if rng > 0 and lower >= 2 * body and upper <= max(body, rng * 0.15):
        pats.append("Hammer")
    for x in range(len(lp)):
        for y in range(x + 1, len(lp)):
            i, j = lp[x], lp[y]
            if (j - i >= 7 and len(Ls) - j <= 30 and abs(lv[y] / lv[x] - 1) <= 0.02
                    and Hs[i:j].max() >= min(lv[x], lv[y]) * 1.04 and close <= lv[y] * 1.05):
                if "Double Bottom" not in pats:
                    pats.append("Double Bottom")
    vol_surge = V[-1] > 1.5 * V[-21:-1].mean() and C[-1] > O[-1]
    if vol_surge:
        pats.append("Volume Surge")
    if a["need_pattern"] and not [p for p in pats if p != "Volume Surge"]:
        return None

    dl = np.diff(C[-15:])
    rsi = 100 - 100 / (1 + max(dl[dl > 0].sum(), 1e-9) / max(-dl[dl < 0].sum(), 1e-9))
    score = touches * 10 + len(pats) * 15 + min(upside, 15) * 2 + min(rr, 4) * 5
    return {"Price": round(close, 2), "Support": round(support, 2),
            "Dist to Support %": round((close / support - 1) * 100, 2),
            "Touches": touches, "Resistance": round(res, 2), "Upside %": round(upside, 2),
            "Stop Loss": round(stop, 2), "R:R": round(rr, 2), "RSI": round(rsi, 1),
            "Patterns": ", ".join(pats) or "-", "Score": round(score, 1)}


# ------------------------------ UI ------------------------------
st.set_page_config(page_title="Support Rebound Screener", layout="wide")
st.title("📈 Nifty 200 Support-Rebound Screener")
st.caption("Finds stocks sitting on a tested support zone with open room (no nearby resistance) to the upside.")

with st.sidebar:
    st.header("Filters")
    a = {
        "dist": st.slider("Max distance above support %", 0.5, 6.0, 3.0, 0.5),
        "upside": st.slider("Min upside to next resistance %", 3.0, 15.0, 6.0, 0.5),
        "rr": st.slider("Min reward : risk", 1.0, 4.0, 1.5, 0.25),
        "need_pattern": st.checkbox("Only show stocks with a bullish pattern", value=False),
    }

if st.button("🔍 Run scan", type="primary"):
    syms = load_universe()
    with st.spinner(f"Downloading {len(syms)} stocks in one batch..."):
        data = yf.download([s + ".NS" for s in syms], period="1y", interval="1d",
                           group_by="ticker", auto_adjust=True, threads=True, progress=False)
    rows = []
    for s in syms:
        try:
            r = analyze(data[s + ".NS"], a)
            if r:
                rows.append({"Stock": s, **r})
        except Exception:
            continue
    st.success(f"Scanned {len(syms)} stocks → {len(rows)} matches")
    if rows:
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download CSV", df.to_csv(index=False), "support_scan.csv")
    else:
        st.info("No matches. Try a larger distance-to-support, lower min upside, or untick the pattern filter.")
    st.caption("How it works: swing-low clusters (≥2 touches) = support; nearest swing high above price = "
               "resistance; stop = 1.5% below support. For education only, not investment advice.")
else:
    st.info("Set filters in the sidebar and click **Run scan**.")
