"""
Seasonality Analysis Dashboard for Yahoo Finance Tickers
--------------------------------------------------------
Run with:
    streamlit run seasonality_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import io

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Seasonality Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Seasonality Analysis Dashboard")
st.caption(
    "Analyze historical seasonality patterns for any Yahoo Finance ticker. "
    "Enter one ticker for full analysis, or several (comma-separated) to compare."
)

# -------------------------------------------------------------------
# Sidebar - User inputs
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

tickers_input = st.sidebar.text_input(
    "Ticker(s)",
    value="SPY",
    help="Enter a single ticker (e.g. AAPL) or comma-separated tickers for comparison (e.g. SPY, QQQ, GLD).",
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    start_date = st.date_input(
        "Start date",
        value=date(2000, 1, 1),
        min_value=date(1950, 1, 1),
        max_value=date.today(),
    )
with col_b:
    end_date = st.date_input(
        "End date",
        value=date.today(),
        min_value=date(1950, 1, 1),
        max_value=date.today(),
    )

price_type = st.sidebar.selectbox(
    "Price type",
    options=["Adj Close", "Close"],
    index=0,
    help="Adjusted Close accounts for splits and dividends.",
)

return_type = st.sidebar.radio(
    "Return type",
    options=["Simple", "Log"],
    index=0,
    horizontal=True,
    help="Simple: (P_t / P_{t-1}) - 1. Log: ln(P_t / P_{t-1}).",
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Yahoo Finance via `yfinance`.")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start, end, price_col: str) -> pd.Series:
    """Download prices from Yahoo Finance and return a clean Series."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    col = price_col if price_col in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
    s = df[col].dropna()
    s.name = ticker
    return s


def compute_returns(prices: pd.Series, kind: str) -> pd.Series:
    if kind == "Log":
        r = np.log(prices / prices.shift(1))
    else:
        r = prices.pct_change()
    return r.dropna()


def monthly_returns(daily_ret: pd.Series, kind: str) -> pd.Series:
    if kind == "Log":
        return daily_ret.resample("ME").sum()
    return (1 + daily_ret).resample("ME").prod() - 1


def quarterly_returns(daily_ret: pd.Series, kind: str) -> pd.Series:
    if kind == "Log":
        return daily_ret.resample("QE").sum()
    return (1 + daily_ret).resample("QE").prod() - 1


def month_year_matrix(m_ret: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with years as rows and months (1..12) as columns."""
    df = m_ret.to_frame("ret")
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    mat = df.pivot(index="Year", columns="Month", values="ret")
    for m in range(1, 13):
        if m not in mat.columns:
            mat[m] = np.nan
    mat = mat[sorted(mat.columns)]
    mat.columns = MONTH_NAMES
    return mat


def summary_stats(mat: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=MONTH_NAMES)
    stats["Avg Return %"] = mat.mean() * 100
    stats["Median %"] = mat.median() * 100
    stats["Std %"] = mat.std() * 100
    stats["Win Rate %"] = (mat > 0).sum() / mat.count() * 100
    stats["Best %"] = mat.max() * 100
    stats["Worst %"] = mat.min() * 100
    stats["N Years"] = mat.count().astype(int)
    return stats


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if not tickers:
    st.info("👈 Enter at least one ticker in the sidebar to begin.")
    st.stop()

# Download data for each ticker
with st.spinner(f"Downloading data for {', '.join(tickers)}..."):
    price_data = {}
    for tk in tickers:
        try:
            s = load_prices(tk, start_date, end_date, price_type)
            if not s.empty:
                price_data[tk] = s
            else:
                st.warning(f"No data returned for **{tk}**. Check the ticker symbol.")
        except Exception as e:
            st.error(f"Failed to download **{tk}**: {e}")

if not price_data:
    st.error("No valid data was retrieved. Please adjust your inputs.")
    st.stop()

# Primary ticker (first one) drives single-ticker views
primary = tickers[0] if tickers[0] in price_data else list(price_data.keys())[0]
prices = price_data[primary]
returns = compute_returns(prices, return_type)
m_ret = monthly_returns(returns, return_type)
q_ret = quarterly_returns(returns, return_type)
mat = month_year_matrix(m_ret)
stats = summary_stats(mat)

# -------------------------------------------------------------------
# Top-level KPI cards
# -------------------------------------------------------------------
current_month_idx = datetime.now().month - 1
current_month_name = MONTH_NAMES[current_month_idx]
best_month = stats["Avg Return %"].idxmax()
worst_month = stats["Avg Return %"].idxmin()
most_consistent = stats["Win Rate %"].idxmax()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Primary ticker", primary)
k2.metric("Best month (avg)", best_month, f"{stats.loc[best_month, 'Avg Return %']:.2f}%")
k3.metric("Worst month (avg)", worst_month, f"{stats.loc[worst_month, 'Avg Return %']:.2f}%")
k4.metric("Most consistent", most_consistent, f"{stats.loc[most_consistent, 'Win Rate %']:.0f}% win rate")
k5.metric(
    f"Current month ({current_month_name})",
    f"{stats.loc[current_month_name, 'Avg Return %']:.2f}%",
    f"Win {stats.loc[current_month_name, 'Win Rate %']:.0f}%",
)

st.markdown("---")

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_over, tab_month, tab_daily, tab_quarter, tab_compare, tab_raw = st.tabs(
    ["Overview", "Monthly", "Weekly / Daily", "Quarterly", "Comparison", "Raw Data"]
)

# ---------------- Overview ----------------
with tab_over:
    st.subheader(f"Price history — {primary}")
    fig_price = px.line(prices, labels={"value": "Price", "index": "Date"})
    fig_price.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Average intra-year seasonality curve")
    st.caption(
        "Average cumulative return path from Jan 1 through Dec 31, "
        "computed across all years in the sample."
    )
    daily_df = returns.to_frame("ret")
    daily_df["Year"] = daily_df.index.year
    daily_df["DOY"] = daily_df.index.dayofyear
    if return_type == "Log":
        daily_df["cum"] = daily_df.groupby("Year")["ret"].cumsum()
    else:
        daily_df["cum"] = daily_df.groupby("Year")["ret"].transform(lambda x: (1 + x).cumprod() - 1)
    avg_curve = daily_df.groupby("DOY")["cum"].mean() * 100

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=avg_curve.index, y=avg_curve.values,
        mode="lines", fill="tozeroy",
        name="Average path",
    ))
    fig_curve.update_layout(
        xaxis_title="Day of year",
        yaxis_title="Average cumulative return (%)",
        height=380, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_curve, use_container_width=True)


# ---------------- Monthly ----------------
with tab_month:
    st.subheader(f"Monthly returns heatmap — {primary}")
    heat_data = mat.copy() * 100  # to %
    avg_row = pd.DataFrame(heat_data.mean(axis=0)).T
    avg_row.index = ["AVG"]
    heat_with_avg = pd.concat([heat_data, avg_row])

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_with_avg.values,
        x=heat_with_avg.columns,
        y=heat_with_avg.index.astype(str),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Return %"),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        text=np.round(heat_with_avg.values, 1),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig_heat.update_layout(
        height=max(400, 22 * len(heat_with_avg)),
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average monthly return (± std)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=stats.index,
            y=stats["Avg Return %"],
            error_y=dict(type="data", array=stats["Std %"], visible=True),
            marker_color=["#2ca02c" if v > 0 else "#d62728" for v in stats["Avg Return %"]],
            text=[f"{v:.2f}%" for v in stats["Avg Return %"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            yaxis_title="Return (%)",
            height=400, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Win rate by month")
        fig_win = go.Figure()
        fig_win.add_trace(go.Bar(
            x=stats.index,
            y=stats["Win Rate %"],
            marker_color="#1f77b4",
            text=[f"{v:.0f}%" for v in stats["Win Rate %"]],
            textposition="outside",
        ))
        fig_win.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50% (coin flip)")
        fig_win.update_layout(
            yaxis_title="Win rate (%)",
            yaxis_range=[0, 105],
            height=400, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_win, use_container_width=True)

    st.subheader("Monthly statistics")
    st.dataframe(
        stats.style.format({
            "Avg Return %": "{:.2f}", "Median %": "{:.2f}", "Std %": "{:.2f}",
            "Win Rate %": "{:.1f}", "Best %": "{:.2f}", "Worst %": "{:.2f}",
        }).background_gradient(subset=["Avg Return %"], cmap="RdYlGn"),
        use_container_width=True,
    )

    csv_buf = io.StringIO()
    stats.to_csv(csv_buf)
    st.download_button(
        "⬇️ Download monthly seasonality (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"{primary}_monthly_seasonality.csv",
        mime="text/csv",
    )


# ---------------- Weekly / Daily ----------------
with tab_daily:
    st.subheader(f"Day-of-week average returns — {primary}")
    dow = returns.copy().to_frame("ret")
    dow["dow"] = dow.index.dayofweek
    dow_mean = dow.groupby("dow")["ret"].mean() * 100
    dow_std = dow.groupby("dow")["ret"].std() * 100
    dow_mean = dow_mean.reindex(range(5))
    dow_std = dow_std.reindex(range(5))
    dow_mean.index = DOW_NAMES
    dow_std.index = DOW_NAMES

    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(
        x=dow_mean.index,
        y=dow_mean.values,
        marker_color=["#2ca02c" if v > 0 else "#d62728" for v in dow_mean.values],
        text=[f"{v:.3f}%" for v in dow_mean.values],
        textposition="outside",
    ))
    fig_dow.update_layout(
        yaxis_title="Average daily return (%)",
        height=380, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    st.subheader("Day-of-month average returns")
    dom = returns.copy().to_frame("ret")
    dom["day"] = dom.index.day
    dom_avg = dom.groupby("day")["ret"].mean() * 100

    fig_dom = go.Figure()
    fig_dom.add_trace(go.Scatter(
        x=dom_avg.index, y=dom_avg.values, mode="lines+markers",
    ))
    fig_dom.add_hline(y=0, line_color="gray")
    fig_dom.update_layout(
        xaxis_title="Day of month",
        yaxis_title="Average daily return (%)",
        height=380, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_dom, use_container_width=True)


# ---------------- Quarterly ----------------
with tab_quarter:
    st.subheader(f"Quarterly seasonality — {primary}")
    qdf = q_ret.to_frame("ret")
    qdf["Q"] = qdf.index.quarter
    q_mean = qdf.groupby("Q")["ret"].mean() * 100
    q_std = qdf.groupby("Q")["ret"].std() * 100
    q_count = qdf.groupby("Q")["ret"].count()
    q_win = qdf.groupby("Q")["ret"].apply(lambda x: (x > 0).mean() * 100)

    q_stats = pd.DataFrame({
        "Avg %": q_mean,
        "Std %": q_std,
        "N Quarters": q_count,
        "Win Rate %": q_win,
    })
    q_stats.index = [f"Q{i}" for i in q_stats.index]

    fig_q = go.Figure()
    fig_q.add_trace(go.Bar(
        x=q_stats.index, y=q_stats["Avg %"],
        error_y=dict(type="data", array=q_stats["Std %"], visible=True),
        marker_color=["#2ca02c" if v > 0 else "#d62728" for v in q_stats["Avg %"]],
        text=[f"{v:.2f}%" for v in q_stats["Avg %"]],
        textposition="outside",
    ))
    fig_q.update_layout(
        yaxis_title="Average quarterly return (%)",
        height=380, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    st.dataframe(
        q_stats.style.format({"Avg %": "{:.2f}", "Std %": "{:.2f}", "Win Rate %": "{:.1f}"}),
        use_container_width=True,
    )


# ---------------- Comparison ----------------
with tab_compare:
    if len(price_data) < 2:
        st.info("Enter at least two comma-separated tickers in the sidebar to compare.")
    else:
        st.subheader("Average monthly return — comparison")
        comp = pd.DataFrame(index=MONTH_NAMES)
        for tk, ps in price_data.items():
            r = compute_returns(ps, return_type)
            mr = monthly_returns(r, return_type)
            mat_tk = month_year_matrix(mr)
            comp[tk] = mat_tk.mean() * 100

        fig_cmp = go.Figure()
        for tk in comp.columns:
            fig_cmp.add_trace(go.Bar(name=tk, x=comp.index, y=comp[tk]))
        fig_cmp.update_layout(
            barmode="group",
            yaxis_title="Average return (%)",
            height=430, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("Comparison table")
        st.dataframe(
            comp.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None),
            use_container_width=True,
        )


# ---------------- Raw data ----------------
with tab_raw:
    st.subheader(f"Monthly return matrix (%) — {primary}")
    st.dataframe(
        (mat * 100).style.format("{:.2f}")
                        .background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True,
    )

    csv_buf2 = io.StringIO()
    (mat * 100).to_csv(csv_buf2)
    st.download_button(
        "⬇️ Download year × month matrix (CSV)",
        data=csv_buf2.getvalue(),
        file_name=f"{primary}_year_month_matrix.csv",
        mime="text/csv",
    )

    st.subheader("Daily returns (last 500 rows)")
    st.dataframe(returns.to_frame("Return").tail(500), use_container_width=True)

st.markdown("---")
st.caption(
    "⚠️ Past performance does not guarantee future results. "
    "This dashboard is for educational purposes only and is not investment advice."
)

