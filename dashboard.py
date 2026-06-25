import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

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
    dp_clean = 0.20
    dp_max = 0.50
    recovery_clean = 82.0
    recovery_min = 60.0

    Pn = (out["deltaP_bar"] - dp_clean) / (dp_max - dp_clean)
    Rn = (recovery_clean - out["acid_recovery_pct"]) / (recovery_clean - recovery_min)

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
# KPI cards
# -----------------------------
st.subheader("Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)

k1.metric("ML Acid Recovery", f"{latest['ml_acid_recovery_pct']:.1f} %")
k2.metric("ML Fouling Score", f"{latest['ml_fouling_score']:.0f} / 100")
k3.metric("Pressure Drop", f"{latest['deltaP_bar']:.2f} bar")
k4.metric("Formula Recovery", f"{latest['acid_recovery_pct']:.1f} %")

st.subheader("Cleaning Forecast")
f1, f2 = st.columns(2)
f1.metric("ML Days to Cleaning", f"{latest['ml_days_to_cleaning']:.2f} days")
f2.metric("Estimated Cleaning Date", estimated_cleaning_date.strftime("%Y-%m-%d %H:%M"))

# -----------------------------
# Gauge charts
# -----------------------------
st.subheader("Live Gauges")
g1, g2, g3 = st.columns(3)

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

with t1:
    st.markdown("**Acid Recovery Trend**")
    st.line_chart(
        df.set_index("time")[["acid_recovery_pct", "ml_acid_recovery_pct"]],
        height=300
    )

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
