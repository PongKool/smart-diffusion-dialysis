import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("diffusion_dialysis_input_data.csv")

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
y_days = df["days_to_cleaning"]

X_train, X_test, yrec_train, yrec_test = train_test_split(X, y_recovery, test_size=0.2, random_state=42)
_, _, yfol_train, yfol_test = train_test_split(X, y_fouling, test_size=0.2, random_state=42)
_, _, yday_train, yday_test = train_test_split(X, y_days, test_size=0.2, random_state=42)

recovery_model = RandomForestRegressor(n_estimators=100, random_state=42)
fouling_model = RandomForestRegressor(n_estimators=100, random_state=42)
days_model = RandomForestRegressor(n_estimators=100, random_state=42)

recovery_model.fit(X_train, yrec_train)
fouling_model.fit(X_train, yfol_train)
days_model.fit(X_train, yday_train)

rec_pred = recovery_model.predict(X_test)
fol_pred = fouling_model.predict(X_test)
day_pred = days_model.predict(X_test)

print("Acid Recovery Model")
print("MAE:", mean_absolute_error(yrec_test, rec_pred))
print("R2 :", r2_score(yrec_test, rec_pred))

print("\nFouling Score Model")
print("MAE:", mean_absolute_error(yfol_test, fol_pred))
print("R2 :", r2_score(yfol_test, fol_pred))

print("\nDays to Cleaning Model")
print("MAE:", mean_absolute_error(yday_test, day_pred))
print("R2 :", r2_score(yday_test, day_pred))

joblib.dump(recovery_model, "recovery_model.pkl")
joblib.dump(fouling_model, "fouling_model.pkl")
joblib.dump(days_model, "days_to_cleaning_model.pkl")

print("\nModels saved:")
print("- recovery_model.pkl")
print("- fouling_model.pkl")
print("- days_to_cleaning_model.pkl")
