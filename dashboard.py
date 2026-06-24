import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# Helper functions
# -----------------------------
def temp_correct_cond(cond, temp, alpha=0.02):
    return cond / (1 + alpha * (temp - 25.0))

def calculate_metrics(df):
    out = df.copy()

    # Temperature-corrected conductivity
    out["feed_cond_25"] = temp_correct_cond(out["S2_feed_cond_mScm"], out["S3_temp_C"])
    out["outlet_cond_25"] = temp_correct_cond(out["S6_outlet_cond_mScm"], out["S3_temp_C"])

    # Formula-based instantaneous acid recovery proxy (%)
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

    # Formula-based membrane state
    def classify_state(score):
        if score <= 20:
            return "Clean"
        elif score <= 40:
            return "Slight fouling"
        elif score <= 60:
            return "Moderate fouling"
        elif score <= 80:
            return "Heavy fouling"
        else:
            return "Severe / Clean now"

    out["membrane_state"] = out["fouling_score"].apply(classify_state)

    return out


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Diffusion Dialysis Dashboard", layout="wide")
st.title("Diffusion Dialysis Pilot Dashboard")
st.caption("7-sensor pilot dashboard with formula-based and ML-based predictions")

# -----------------------------
# Load CSV data
# -----------------------------
csv_file = "diffusion_dialysis_input_data.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"File not found: {csv_file}")
    st.stop()

df["time"] = pd.to_datetime(df["time"])

# -----------------------------
# Calculate formula-based outputs
# -----------------------------
df = calculate_metrics(df)

# -----------------------------
# Load ML models
# -----------------------------
try:
    recovery_model = joblib.load("recovery_model.pkl")
    fouling_model = joblib.load("fouling_model.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# -----------------------------
# ML predictions
# -----------------------------
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

df["ml_acid_recovery_pct"] = recovery_model.predict(X)
df["ml_fouling_score"] = fouling_model.predict(X)

# ML membrane state
def classify_ml_state(score):
    if score <= 20:
        return "Clean"
    elif score <= 40:
        return "Slight fouling"
    elif score <= 60:
        return "Moderate fouling"
    elif score <= 80:
        return "Heavy fouling"
    else:
        return "Severe / Clean now"

df["ml_membrane_state"] = df["ml_fouling_score"].apply(classify_ml_state)

# Latest row
latest = df.iloc[-1]

# -----------------------------
# KPI section
# -----------------------------
st.subheader("Current Status")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Formula Acid Recovery (%)", f"{latest['acid_recovery_pct']:.1f}")
c2.metric("ML Acid Recovery (%)", f"{latest['ml_acid_recovery_pct']:.1f}")
c3.metric("Formula Fouling Score", f"{latest['fouling_score']:.0f} / 100")
c4.metric("ML Fouling Score", f"{latest['ml_fouling_score']:.0f} / 100")

c5, c6 = st.columns(2)
c5.metric("Pressure Drop (bar)", f"{latest['deltaP_bar']:.2f}")
c6.metric("ML Membrane State", latest["ml_membrane_state"])

# -----------------------------
# Recommendation section
# -----------------------------
st.subheader("Recommended Action (Based on ML Fouling Score)")

if latest["ml_fouling_score"] > 80:
    st.error("Cleaning recommended now")
elif latest["ml_fouling_score"] > 60:
    st.warning("Prepare cleaning")
else:
    st.success("Normal operation")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Prediction Trends")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Acid Recovery: Formula vs ML**")
    st.line_chart(
        df.set_index("time")[["acid_recovery_pct", "ml_acid_recovery_pct"]],
        height=300
    )

with col2:
    st.markdown("**Fouling Score: Formula vs ML**")
    st.line_chart(
        df.set_index("time")[["fouling_score", "ml_fouling_score"]],
        height=300
    )

st.subheader("Pressure Trend")
st.line_chart(
    df.set_index("time")[["S4_inlet_press_bar", "S5_outlet_press_bar", "deltaP_bar"]],
    height=300
)

st.subheader("Sensor Trends")
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
# Data table
# -----------------------------
st.subheader("Data Table")

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
    "membrane_state",
    "ml_membrane_state"
]

st.dataframe(df[display_cols], use_container_width=True)

# -----------------------------
# Demo cleaning control
# -----------------------------
st.subheader("Operator Control")

if st.button("Start Cleaning"):
    st.info("Cleaning sequence initiated (demo only)")
