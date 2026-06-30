import math
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go

# -----------------------------
# Global System Constraints
# -----------------------------
DP_CLEAN = 0.20
DP_MAX = 0.50
RECOVERY_CLEAN = 82.0
RECOVERY_MIN = 60.0

# -----------------------------
# Helper functions
# -----------------------------
def temp_correct_cond(cond, temp, alpha=0.02):
    return cond / (1 + alpha * (temp - 25.0))

def calculate_metrics(df):
    out = df.copy()

    out["feed_cond_25"] = temp_correct_cond(out["S2_feed_cond_mScm"], out["S3_temp_C"])
    out["outlet_cond_25"] = temp_correct_cond(out["S6_outlet_cond_mScm"], out["S3_temp_C"])

    # Formula-based acid recovery proxy
    out["acid_recovery_pct"] = (
        (out["S7_water_flow_Lmin"] * out["outlet_cond_25"]) /
        (out["S1_feed_flow_Lmin"] * out["feed_cond_25"])
    ) * 100

    # Pressure drop
    out["deltaP_bar"] = out["S4_inlet_press_bar"] - out["S5_outlet_press_bar"]

    # Formula-based fouling score
    Pn = (out["deltaP_bar"] - DP_CLEAN) / (DP_MAX - DP_CLEAN)
    Rn = (RECOVERY_CLEAN - out["acid_recovery_pct"]) / (RECOVERY_CLEAN - RECOVERY_MIN)

    Pn = Pn.clip(0, 1)
    Rn = Rn.clip(0, 1)

    FI = 0.5 * Pn + 0.5 * Rn
    out["fouling_score"] = (1 + 99 * FI).clip(1, 100)

    return out

def classify_state(score):
    if score <= 20:
        return "Clean"
    elif score <= 40:
        return "Slight Fouling"
    elif score <= 60:
        return "Moderate Fouling"
    elif score <= 80:
        return "Heavy Fouling"
    else:
        return "Severe / Clean Now"

def get_status_color(score):
    if score <= 20:
        return "#16a34a"   # green
    elif score <= 40:
        return "#84cc16"   # light green
    elif score <= 60:
        return "#f59e0b"   # amber
    elif score <= 80:
        return "#f97316"   # orange
    else:
        return "#dc2626"   # red

def gauge_chart(value, title, min_val=0, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 20], "color": "#dcfce7"},
                {"range": [20, 40], "color": "#d9f99d"},
                {"range": [40, 60], "color": "#fde68a"},
                {"range": [60, 80], "color": "#fdba74"},
                {"range": [80, 100], "color": "#fecaca"},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def pressure_gauge(value, title, min_val=0, max_val=1.0):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        title={"text": title},
        number={"suffix": " bar"},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "#0f766e"},
            "steps": [
                {"range": [0, 0.25], "color": "#dcfce7"},
                {"range": [0.25, 0.40], "color": "#fde68a"},
                {"range": [0.40, 1.0], "color": "#fecaca"},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def make_days_to_cleaning_gauge(days_value):
    # Auto-scale gauge max
    gauge_max = max(5.0, math.ceil(float(days_value) + 1.0))

    if days_value <= 1:
        forecast_gauge_color = "#dc2626"  # red
    elif days_value <= 3:
        forecast_gauge_color = "#f59e0b"  # amber
    else:
        forecast_gauge_color = "#2563eb"  # blue

    # Dynamic step boundaries
    step1_end = min(1.0, gauge_max)
    step2_end = min(3.0, gauge_max)

    steps = []
    if step1_end > 0:
        steps.append({"range": [0, step1_end], "color": "#fecaca"})   # critical
    if step2_end > step1_end:
        steps.append({"range": [step1_end, step2_end], "color": "#fde68a"})  # warning
    if gauge_max > step2_end:
        steps.append({"range": [step2_end, gauge_max], "color": "#dbeafe"})  # normal

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(days_value),
        title={"text": "Days to Cleaning"},
        number={"suffix": " d"},
        gauge={
            "axis": {"range": [0, gauge_max]},
            "bar": {"color": forecast_gauge_color},
            "steps": steps,
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Diffusion Dialysis Dashboard", layout="wide")
st.title("Diffusion Dialysis Pilot Dashboard")
st.caption("ML-enhanced digital twin demo for acid recovery, membrane fouling, and days to cleaning")

# -----------------------------
# Load data
# -----------------------------
csv_file = "diffusion_dialysis_input_data.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"File not found: {csv_file}")
    st.stop()

df["time"] = pd.to_datetime(df["time"])
df = calculate_metrics(df)

# -----------------------------
# Load ML models
# -----------------------------
try:
    recovery_model = joblib.load("recovery_model.pkl")
    fouling_model = joblib.load("fouling_model.pkl")
    days_model = joblib.load("days_to_cleaning_model.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

feature_cols = [
    "S1_feed_flow_Lmin",
    "S2_feed_cond_mScm",
    "S3_temp_C",
    "S4_inlet_press_bar",
    "S7_water_flow_Lmin",
    "S5_outlet_press_bar",
    "S6_outlet_cond_mScm"
]

X = df[feature_cols]

# -----------------------------
# ML predictions
# -----------------------------
df["ml_acid_recovery_pct"] = recovery_model.predict(X)
df["ml_fouling_score"] = fouling_model.predict(X)
df["ml_days_to_cleaning"] = days_model.predict(X).clip(min=0)
df["ml_membrane_state"] = df["ml_fouling_score"].apply(classify_state)

latest = df.iloc[-1]
status_color = get_status_color(latest["ml_fouling_score"])
estimated_cleaning_date = latest["time"] + pd.to_timedelta(latest["ml_days_to_cleaning"], unit="D")
last_update_time = latest["time"].strftime("%Y-%m-%d %H:%M")
membrane_health_pct = max(0, 100 - latest["ml_fouling_score"])

# -----------------------------
# Status banner
# -----------------------------
st.markdown(
    f"""
    <div style="
        background-color:{status_color};
        padding:18px;
        border-radius:12px;
        color:white;
        font-size:24px;
        font-weight:700;
        text-align:center;
        margin-bottom:18px;">
        Current Membrane Status: {latest["ml_membrane_state"]}
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Last update time
# -----------------------------
st.markdown(
    f"""
    <div style="
        background-color:#f3f4f6;
        padding:10px 14px;
        border-radius:10px;
        color:#111827;
        font-size:16px;
        margin-bottom:16px;">
        <b>Last Update Time:</b> {last_update_time}
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# KPI cards
# -----------------------------
st.subheader("Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)

k1.metric("ML Acid Recovery", f"{latest['ml_acid_recovery_pct']:.1f} %")
k2.metric("ML Fouling Score", f"{latest['ml_fouling_score']:.0f} / 100")
k3.metric("Pressure Drop", f"{latest['deltaP_bar']:.2f} bar")
k4.metric("Formula Recovery", f"{latest['acid_recovery_pct']:.1f} %")

# -----------------------------
# Cleaning Forecast Box
# -----------------------------
if latest["ml_days_to_cleaning"] <= 1:
    forecast_color = "#dc2626"
    forecast_text = "Cleaning Required Soon"
elif latest["ml_days_to_cleaning"] <= 3:
    forecast_color = "#f59e0b"
    forecast_text = "Prepare for Cleaning"
else:
    forecast_color = "#2563eb"
    forecast_text = "Cleaning Forecast Normal"

st.markdown(
    f"""
    <div style="
        background-color:{forecast_color};
        padding:20px;
        border-radius:14px;
        color:white;
        margin-top:10px;
        margin-bottom:18px;">
        <div style="font-size:24px; font-weight:700; margin-bottom:8px;">
            Cleaning Forecast
        </div>
        <div style="font-size:18px; margin-bottom:6px;">
            Status: <b>{forecast_text}</b>
        </div>
        <div style="font-size:18px; margin-bottom:6px;">
            ML Days to Cleaning: <b>{latest['ml_days_to_cleaning']:.2f} days</b>
        </div>
        <div style="font-size:18px;">
            Estimated Cleaning Date: <b>{estimated_cleaning_date.strftime("%Y-%m-%d %H:%M")}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Membrane health progress bar
# -----------------------------
st.subheader("Membrane Health")

if membrane_health_pct >= 70:
    health_text = "Healthy"
    health_color = "#16a34a"
elif membrane_health_pct >= 40:
    health_text = "Degrading"
    health_color = "#f59e0b"
else:
    health_text = "Poor / Near Cleaning"
    health_color = "#dc2626"

st.markdown(
    f"""
    <div style="
        background-color:#f9fafb;
        padding:16px;
        border-radius:12px;
        margin-bottom:12px;">
        <div style="font-size:20px; font-weight:700; color:#111827; margin-bottom:6px;">
            Membrane Health
        </div>
        <div style="font-size:17px; color:{health_color}; margin-bottom:8px;">
            <b>{health_text}</b> — {membrane_health_pct:.1f}%
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.progress(membrane_health_pct / 100.0)

# -----------------------------
# Gauge charts
# -----------------------------
st.subheader("Live Gauges")
g1, g2, g3, g4 = st.columns(4)

with g1:
    st.plotly_chart(
        gauge_chart(latest["ml_acid_recovery_pct"], "ML Acid Recovery (%)", 0, 100),
        use_container_width=True
    )

with g2:
    st.plotly_chart(
        gauge_chart(latest["ml_fouling_score"], "ML Fouling Score", 0, 100),
        use_container_width=True
    )

with g3:
    st.plotly_chart(
        pressure_gauge(latest["deltaP_bar"], "Pressure Drop", 0, 1.0),
        use_container_width=True
    )

with g4:
    st.plotly_chart(
        make_days_to_cleaning_gauge(latest["ml_days_to_cleaning"]),
        use_container_width=True
    )

# -----------------------------
# Recommendation box
# -----------------------------
st.subheader("Recommended Action")

if latest["ml_fouling_score"] > 80:
    st.error("Immediate cleaning recommended.")
elif latest["ml_fouling_score"] > 60:
    st.warning("Prepare for cleaning. Fouling is significant.")
elif latest["ml_fouling_score"] > 40:
    st.info("Monitor membrane condition closely.")
else:
    st.success("System operating normally.")

if latest["ml_days_to_cleaning"] <= 1:
    st.error("Predicted cleaning need within 1 day.")
elif latest["ml_days_to_cleaning"] <= 3:
    st.warning("Predicted cleaning need within 3 days.")
else:
    st.info(f"Estimated remaining time before cleaning: {latest['ml_days_to_cleaning']:.2f} days.")

# -----------------------------
# Trend charts
# -----------------------------
st.subheader("Trend Analysis")
t1, t2 = st.columns(2)

# -----------------------------
# 7-Day Acid Recovery Projection
# -----------------------------
with t1:
    st.markdown("**Acid Recovery Trend & 7-Day Projection**")
    
    # 1. Prepare historical data
    hist_df = df[["time", "acid_recovery_pct", "ml_acid_recovery_pct"]].copy()
    last_time = hist_df["time"].max()
    latest_ml_rec = hist_df["ml_acid_recovery_pct"].iloc[-1]
    
    # 2. Get days remaining and define the physical floor
    days_remaining = float(latest["ml_days_to_cleaning"])
    
    # 3. Calculate the real degradation rate per day to reach the floor
    # Dynamically derive degradation based on real global thresholds
    
    if days_remaining > 0:
        daily_drop_rate = (latest_ml_rec - RECOVERY_MIN / days_remaining
    else:
        daily_drop_rate = 5.0  # Aggressive drop if already overdue
        
    # 4. Generate future timestamps (next 7 days)
    future_times = pd.date_range(start=last_time, periods=8, freq="D")[1:]
    
    future_preds = []
    for i in range(1, 8):
        # Linearly degrade performance based on the calculated drop rate
        pred = latest_ml_rec - (i * daily_drop_rate)
        # Prevent the plot from dropping below a realistic absolute floor
        future_preds.append(max(55.0, pred))
        
    # 5. Create projection DataFrame
    projection_df = pd.DataFrame({
        "time": future_times,
        "7-Day Projected Recovery": future_preds
    })
    
    # Connect the last historical point to the first projection point
    projection_df = pd.concat([
        pd.DataFrame({
            "time": [last_time], 
            "7-Day Projected Recovery": [latest_ml_rec]
        }), 
        projection_df
    ], ignore_index=True)
    
    # 6. Combine and rename for display
    chart_df = pd.merge(hist_df[["time", "acid_recovery_pct", "ml_acid_recovery_pct"]], projection_df, on="time", how="outer")
    chart_df = chart_df.rename(columns={
        "acid_recovery_pct": "Formula Recovery",
        "ml_acid_recovery_pct": "Historical ML Recovery"
    })
    
    st.line_chart(chart_df.set_index("time"), height=300)
    

with t2:
    st.markdown("**Fouling Score Trend**")
    st.line_chart(
        df.set_index("time")[["fouling_score", "ml_fouling_score"]],
        height=300
    )

st.markdown("**Pressure Trend**")
st.line_chart(
    df.set_index("time")[["S4_inlet_press_bar", "S5_outlet_press_bar", "deltaP_bar"]],
    height=300
)

st.markdown("**Days to Cleaning Trend**")
st.line_chart(
    df.set_index("time")[["ml_days_to_cleaning"]],
    height=300
)

st.markdown("**Sensor Trend Overview**")
st.line_chart(
    df.set_index("time")[[
        "S1_feed_flow_Lmin",
        "S7_water_flow_Lmin",
        "S2_feed_cond_mScm",
        "S6_outlet_cond_mScm",
        "S3_temp_C"
    ]],
    height=300
)

# -----------------------------
# Current sensor snapshot
# -----------------------------
st.subheader("Current Sensor Snapshot")
s1, s2, s3, s4 = st.columns(4)

s1.metric("Feed Flow", f"{latest['S1_feed_flow_Lmin']:.2f} L/min")
s2.metric("Water Flow", f"{latest['S7_water_flow_Lmin']:.2f} L/min")
s3.metric("Feed Conductivity", f"{latest['S2_feed_cond_mScm']:.1f} mS/cm")
s4.metric("Outlet Conductivity", f"{latest['S6_outlet_cond_mScm']:.1f} mS/cm")

s5, s6, s7 = st.columns(3)
s5.metric("Temperature", f"{latest['S3_temp_C']:.1f} °C")
s6.metric("Inlet Pressure", f"{latest['S4_inlet_press_bar']:.2f} bar")
s7.metric("Outlet Pressure", f"{latest['S5_outlet_press_bar']:.2f} bar")

# -----------------------------
# Data table
# -----------------------------
with st.expander("Show Data Table"):
    display_cols = [
        "time",
        "S1_feed_flow_Lmin",
        "S2_feed_cond_mScm",
        "S3_temp_C",
        "S4_inlet_press_bar",
        "S7_water_flow_Lmin",
        "S5_outlet_press_bar",
        "S6_outlet_cond_mScm",
        "deltaP_bar",
        "acid_recovery_pct",
        "ml_acid_recovery_pct",
        "fouling_score",
        "ml_fouling_score",
        "ml_days_to_cleaning",
        "ml_membrane_state"
    ]
    st.dataframe(df[display_cols], use_container_width=True)

# -----------------------------
# Operator control
# -----------------------------
st.subheader("Operator Control")
b1, b2 = st.columns(2)

with b1:
    if st.button("Start Cleaning"):
        st.info("Cleaning sequence initiated (demo only).")

with b2:
    if st.button("Acknowledge Alert"):
        st.success("Alert acknowledged.")
