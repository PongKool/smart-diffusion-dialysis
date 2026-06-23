import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # <-- Add this line

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

    # Instantaneous acid recovery proxy (%)
    out["acid_recovery_pct"] = (
        (out["S7_water_flow_Lmin"] * out["outlet_cond_25"]) /
        (out["S1_feed_flow_Lmin"] * out["feed_cond_25"])
    ) * 100

    # Pressure drop
    out["deltaP_bar"] = out["S4_inlet_press_bar"] - out["S5_outlet_press_bar"]

    # Fouling score
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
# Streamlit page
# -----------------------------
st.set_page_config(page_title="Diffusion Dialysis Dashboard", layout="wide")
st.title("Diffusion Dialysis Pilot Dashboard")
st.caption("7-sensor pilot dashboard for acid recovery and fouling monitoring")

# Load CSV
csv_file = "diffusion_dialysis_input_data.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"File not found: {csv_file}")
    st.stop()

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# Calculate dashboard metrics
df = calculate_metrics(df)

# Latest row
latest = df.iloc[-1]

# -----------------------------
# KPI cards
# -----------------------------
st.subheader("Current Status")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Acid Recovery (%)", f"{latest['acid_recovery_pct']:.1f}")
c2.metric("Fouling Score", f"{latest['fouling_score']:.0f} / 100")
c3.metric("Pressure Drop (bar)", f"{latest['deltaP_bar']:.2f}")
c4.metric("Membrane State", latest["membrane_state"])

# -----------------------------
# Action recommendation
# -----------------------------
st.subheader("Recommended Action")

if latest["fouling_score"] > 80:
    st.error("Cleaning recommended now")
elif latest["fouling_score"] > 60:
    st.warning("Prepare cleaning")
else:
    st.success("Normal operation")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Performance Trends")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Acid Recovery and Fouling Score**")
    st.line_chart(
        df.set_index("time")[["acid_recovery_pct", "fouling_score"]],
        height=300
    )

with col2:
    st.markdown("**Pressure Trend**")
    st.line_chart(
        df.set_index("time")[["S4_inlet_press_bar", "S5_outlet_press_bar", "deltaP_bar"]],
        height=300
    )

st.subheader("Flow Rates & Temperature")

# Create a figure with a secondary y-axis
fig = go.Figure()

# Add Flow Rates on the Primary (Left) Y-Axis
fig.add_trace(go.Scatter(
    x=df["time"], 
    y=df["S1_feed_flow_Lmin"], 
    name="S1 Feed Flow (L/min)",
    yaxis="y1"
))
fig.add_trace(go.Scatter(
    x=df["time"], 
    y=df["S7_water_flow_Lmin"], 
    name="S7 Water Flow (L/min)",
    yaxis="y1"
))

# Add Temperature on the Secondary (Right) Y-Axis
fig.add_trace(go.Scatter(
    x=df["time"], 
    y=df["S3_temp_C"], 
    name="S3 Temp (°C)",
    yaxis="y2"
))

# Configure the dual axis layout
fig.update_layout(
    xaxis=dict(title="Time"),
    yaxis=dict(title="Flow Rate (L/min)"),
    yaxis2=dict(
        title="Temperature (°C)",
        overlaying="y",
        side="right"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=40, b=40),
    height=350
)

# Render the interactive chart in Streamlit
st.plotly_chart(fig, use_container_width=True)


st.subheader("Conductivity Trends")
st.line_chart(df.set_index("time")[["S2_feed_cond_mScm", "S6_outlet_cond_mScm"]], height=250)

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
    "fouling_score",
    "membrane_state"
]

st.dataframe(df[display_cols], use_container_width=True)

# -----------------------------
# Demo cleaning control
# -----------------------------
st.subheader("Operator Control")

if st.button("Start Cleaning"):
    st.info("Cleaning sequence initiated (demo only)")
