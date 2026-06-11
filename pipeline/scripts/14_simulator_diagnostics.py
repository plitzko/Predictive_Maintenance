"""
14_simulator_diagnostics.py
Diagnose-Plots fuer den Simulator-Datensatz.

Beantwortet die Frage: machen die simulierten Daten Sinn?
- Health-Score-Trajektorien pro LKW
- Sensor-Drift entlang Health-Score
- Korrelation Route -> Beladung -> Verschleiss
- Faulty-Anteil pro LKW und ueber Zeit
- DTC-Verteilung
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA = Path("data/engine_health/engine_data_simulated.csv")
OUT = Path("outputs/charts")
OUT.mkdir(parents=True, exist_ok=True)

print("=== Running 14_simulator_diagnostics.py ===")
df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Loaded {len(df):,} rows from {DATA}")

sns.set_theme(style="whitegrid")
trucks = sorted(df["truck_id"].unique())
truck_palette = sns.color_palette("husl", n_colors=len(trucks))


# ----------------------------------------------------------
# 1) Health-Score-Trajektorien pro LKW
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
for tid, color in zip(trucks, truck_palette):
    sub = df[df["truck_id"] == tid].sort_values("timestamp")
    # Tagesmittel um Rauschen zu glaetten
    daily = sub.set_index("timestamp")["health_score"].resample("D").mean()
    ax.plot(daily.index, daily.values, label=tid, color=color, linewidth=1.6)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7,
           label="Schadensschwelle 0.5")
ax.set_xlabel("Datum")
ax.set_ylabel("Health-Score (Tagesmittel)")
ax.set_title("Verschleissverlauf pro LKW – simulierte Trajektorien",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower left", ncol=2, fontsize=8)
ax.set_ylim(0, 1.02)
plt.tight_layout()
out = OUT / "13_health_score_trajectories.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {out}")


# ----------------------------------------------------------
# 2) Sensor-Drift: Coolant temp vs Health-Score
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
sensors = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp",
]
sample = df.sample(n=min(8000, len(df)), random_state=42)
for ax, sensor in zip(axes.flat, sensors):
    sc = ax.scatter(
        sample["health_score"], sample[sensor],
        c=sample["load_pct"], cmap="viridis",
        s=4, alpha=0.4,
    )
    ax.set_xlabel("Health-Score")
    ax.set_ylabel(sensor)
    ax.invert_xaxis()
fig.colorbar(sc, ax=axes.ravel().tolist(), label="Beladung %", shrink=0.7)
fig.suptitle("Sensor-Drift: Werte als Funktion des Health-Scores "
             "(Farbe = Beladung)", fontsize=13, fontweight="bold")
out = OUT / "14_sensor_drift.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {out}")


# ----------------------------------------------------------
# 3) Route -> Beladung Boxplot (kausale Kopplung sichtbar?)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="route_type", y="load_pct",
            order=["Stadtverkehr", "Landstrasse", "Autobahn"],
            ax=axes[0], palette="Blues")
axes[0].set_title("Beladung nach Routentyp\n(Autobahn typischerweise höher)")
axes[0].set_ylabel("Beladung %")

# Verschleiss pro Route: durchschnittlicher Health-Score-Verlust pro LKW pro Route
df_sorted = df.sort_values(["truck_id", "timestamp"]).reset_index(drop=True)
df_sorted["d_health"] = df_sorted.groupby("truck_id")["health_score"].diff()
wear_per_route = (
    df_sorted.dropna(subset=["d_health"])
    .groupby("route_type")["d_health"]
    .mean() * -1 * 48
)  # *48 Messungen pro Tag -> Verlust pro Tag
order = ["Stadtverkehr", "Landstrasse", "Autobahn"]
wear_per_route = wear_per_route.reindex(order)
axes[1].bar(wear_per_route.index, wear_per_route.values,
            color=["#1976D2", "#43A047", "#E53935"])
axes[1].set_title("Mittlerer Health-Score-Verlust pro Tag\nje Routentyp")
axes[1].set_ylabel("Health-Score-Verlust / Tag")
for i, v in enumerate(wear_per_route.values):
    axes[1].text(i, v + 1e-4, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10)

plt.tight_layout()
out = OUT / "15_route_load_wear.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {out}")


# ----------------------------------------------------------
# 4) Faulty-Anteil pro LKW + Time-to-Failure-Verteilung
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

faulty_share = df.groupby("truck_id")["Engine Condition"].apply(
    lambda s: (s == 0).mean() * 100
).sort_values()
axes[0].barh(faulty_share.index, faulty_share.values,
             color=["#43A047" if v < 20 else "#FB8C00" if v < 50 else "#E53935"
                    for v in faulty_share.values])
axes[0].set_xlabel("Faulty-Anteil [%]")
axes[0].set_title("Faulty-Anteil pro LKW")
axes[0].axvline(35.8, color="black", linestyle="--", linewidth=1,
                label="Mittelwert 35,8%")
axes[0].legend()
for i, v in enumerate(faulty_share.values):
    axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

# Time-to-First-Failure pro LKW (in Tagen)
ttfs = []
for tid in trucks:
    sub = df[df["truck_id"] == tid].sort_values("timestamp")
    fail = sub[sub["Engine Condition"] == 0]
    if len(fail) == 0:
        ttfs.append((tid, 999))  # zensiert
    else:
        ts = fail["timestamp"].iloc[0] - sub["timestamp"].iloc[0]
        ttfs.append((tid, ts.total_seconds() / 86400))
ttfs_df = pd.DataFrame(ttfs, columns=["truck_id", "ttf_days"])
ttfs_df = ttfs_df[ttfs_df["ttf_days"] < 999].sort_values("ttf_days")

axes[1].barh(ttfs_df["truck_id"], ttfs_df["ttf_days"],
             color="#1976D2")
axes[1].set_xlabel("Tage bis zum ersten Faulty-Zustand")
axes[1].set_title("Time-to-First-Failure pro LKW")
for i, (_, row) in enumerate(ttfs_df.iterrows()):
    axes[1].text(row["ttf_days"] + 1, i,
                 f"{row['ttf_days']:.1f}d", va="center", fontsize=9)

plt.tight_layout()
out = OUT / "16_faulty_distribution.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {out}")


# ----------------------------------------------------------
# 5) DTC-Verteilung im simulierten Datensatz
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

dtc_total = df.loc[df["primary_dtc_code"] != "", "primary_dtc_code"].value_counts()
axes[0].barh(dtc_total.index, dtc_total.values, color="#E53935")
axes[0].set_xlabel("Anzahl Vorkommen")
axes[0].set_title("DTC-Verteilung gesamt")
for i, v in enumerate(dtc_total.values):
    axes[0].text(v + 100, i, f"{v:,}", va="center", fontsize=9)

# DTC-Anteil pro Engine-Condition
dtc_by_cond = df.groupby(["Engine Condition", "primary_dtc_code"]).size().unstack(fill_value=0)
dtc_by_cond = dtc_by_cond.drop(columns=[""], errors="ignore")
dtc_by_cond.T.plot.bar(ax=axes[1], color=["#E53935", "#43A047"], edgecolor="black")
axes[1].set_xlabel("DTC-Code")
axes[1].set_ylabel("Anzahl")
axes[1].set_title("DTC nach Engine-Condition\n(0 = Faulty, 1 = Healthy)")
axes[1].legend(["Faulty (0)", "Healthy (1)"])
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
out = OUT / "17_dtc_distribution_simulated.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {out}")

print("\n=== Done. ===")
