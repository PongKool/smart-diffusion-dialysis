import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Helper functions
# -----------------------------
def temp_correct_cond(cond, temp, alpha=0.02):
    return cond / (1 + alpha * (temp - 25.0))

def calculate_targets(df):
    out = df.copy()

    out["feed_cond_25"] = temp_correct_cond(out["S2_feed_cond_mScm"], out["S3_temp_C"])
    out["outlet_cond_25"] = temp_correct_cond(out["S6_outlet_cond_mScm"], out["S3_temp_C"])

    # Acid recovery proxy
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

    return out

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("diffusion_dialysis_input_data.csv")
df = calculate_targets(df)

# -----------------------------
# Features and targets
# -----------------------------
features = [
    "S1_feed_flow_Lmin",
    "S2_feed_cond_mScm",
    "S3_temp_C",
    "S4_inlet_press_bar",
    "S7_water_flow_Lmin",
    "S5_outlet_press_bar",
    "S6_outlet_cond_mScm"
]

X = df[features]

y_recovery = df["acid_recovery_pct"]
y_fouling = df["fouling_score"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, yrec_train, yrec_test = train_test_split(
    X, y_recovery, test_size=0.2, random_state=42
)

_, _, yfol_train, yfol_test = train_test_split(
    X, y_fouling, test_size=0.2, random_state=42
)

# -----------------------------
# Train models
# -----------------------------
recovery_model = RandomForestRegressor(n_estimators=100, random_state=42)
fouling_model = RandomForestRegressor(n_estimators=100, random_state=42)

recovery_model.fit(X_train, yrec_train)
fouling_model.fit(X_train, yfol_train)

# -----------------------------
# Evaluate
# -----------------------------
rec_pred = recovery_model.predict(X_test)
fol_pred = fouling_model.predict(X_test)

print("Acid Recovery Model")
print("MAE:", mean_absolute_error(yrec_test, rec_pred))
print("R2 :", r2_score(yrec_test, rec_pred))

print("\nFouling Score Model")
print("MAE:", mean_absolute_error(yfol_test, fol_pred))
print("R2 :", r2_score(yfol_test, fol_pred))

# -----------------------------
# Save models
# -----------------------------
joblib.dump(recovery_model, "recovery_model.pkl")
joblib.dump(fouling_model, "fouling_model.pkl")

print("\nModels saved:")
print("- recovery_model.pkl")
print("- fouling_model.pkl")
