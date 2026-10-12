# analyze.py
# Breakdown-risk analysis for Vossberg Mobility fleet.
#
# Key finding: km_since_service, avg_daily_km, and load_factor are the strongest predictors
# of breakdown. Total odometer and age_years show near-zero separation between groups and are
# NOT useful predictors — high-mileage and older cars break down at the same rate as others.
#
# Risk score: a weighted sum of the three predictive factors, each normalised to [0, 1] across
# the fleet, then scaled to [0, 100]. Weights reflect correlation strength with broke_down.

import pandas as pd

df = pd.read_csv("fleet_history.csv")

# --- 1. Show how the groups differ column by column ---
predictors = ["km_since_service", "avg_daily_km", "load_factor", "odometer_km", "age_years"]

print("Mean values by breakdown outcome")
print("-" * 60)
broke = df[df["broke_down"] == 1]
ok    = df[df["broke_down"] == 0]
print(f"{'column':<22} {'broke_down=1':>13} {'broke_down=0':>13} {'diff %':>8}")
for col in predictors:
    m1 = broke[col].mean()
    m0 = ok[col].mean()
    diff_pct = (m1 - m0) / m0 * 100 if m0 else 0.0
    print(f"  {col:<20} {m1:>13.2f} {m0:>13.2f} {diff_pct:>7.1f}%")

print()
corr = df[predictors + ["broke_down"]].corr()["broke_down"].drop("broke_down")
print("Pearson correlation with broke_down (sorted by absolute value):")
for col, val in corr.abs().sort_values(ascending=False).items():
    sign = "+" if corr[col] >= 0 else "-"
    print(f"  {col:<22} r = {sign}{abs(val):.3f}")

# --- 2. Build risk score ---
# Normalise the three predictive columns to [0, 1].
# Weights roughly proportional to their correlation magnitudes: 0.55 / 0.25 / 0.20.
WEIGHTS = {
    "km_since_service": 0.55,
    "avg_daily_km":     0.25,
    "load_factor":      0.20,
}

risk = pd.Series(0.0, index=df.index)
for col, w in WEIGHTS.items():
    col_min = df[col].min()
    col_max = df[col].max()
    normalised = (df[col] - col_min) / (col_max - col_min)
    risk += w * normalised

df["risk_score"] = (risk * 100).round(1)

# --- 3. Print ranked list ---
print()
print("Fleet ranked by breakdown risk (highest first)")
print("-" * 60)
ranked = df[["car_id", "risk_score", "km_since_service", "avg_daily_km", "load_factor", "broke_down"]].sort_values(
    "risk_score", ascending=False
)
print(ranked.to_string(index=False))

print()
print(f"Cars in top-25% risk tier:  {(df['risk_score'] >= df['risk_score'].quantile(0.75)).sum()}")
print(f"Actual breakdowns in that tier: {df[df['risk_score'] >= df['risk_score'].quantile(0.75)]['broke_down'].sum()}")
print(f"Actual breakdowns outside that tier: {df[df['risk_score'] < df['risk_score'].quantile(0.75)]['broke_down'].sum()}")
