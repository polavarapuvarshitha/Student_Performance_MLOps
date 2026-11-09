import os
import json
import pandas as pd

from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split


# File paths
DATA_PATH = "data/StudentsPerformance (1).csv"

OUTPUT_DIR = "outputs/drift"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate reference and current datasets for demonstration
reference, current = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

# Features
numerical_features = [
    "reading score",
    "writing score"
]

categorical_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

results = []


# Numerical drift using KS Test
for column in numerical_features:

    statistic, p_value = ks_2samp(
        reference[column],
        current[column]
    )

    drift = p_value < 0.05

    results.append({
        "feature": column,
        "feature_type": "Numerical",
        "drift_score": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": bool(drift)
    })


# Categorical drift using Jensen-Shannon Distance
for column in categorical_features:

    ref_dist = reference[column].value_counts(normalize=True)
    cur_dist = current[column].value_counts(normalize=True)

    categories = sorted(
        set(ref_dist.index).union(set(cur_dist.index))
    )

    ref_values = ref_dist.reindex(
        categories, fill_value=0
    ).values

    cur_values = cur_dist.reindex(
        categories, fill_value=0
    ).values

    drift_score = jensenshannon(
        ref_values,
        cur_values
    )

    drift = drift_score > 0.1

    results.append({
        "feature": column,
        "feature_type": "Categorical",
        "drift_score": round(float(drift_score), 4),
        "p_value": None,
        "drift_detected": bool(drift)
    })


# Create report
report = pd.DataFrame(results)

report.to_csv(
    os.path.join(OUTPUT_DIR, "data_drift_report.csv"),
    index=False
)

with open(
    os.path.join(OUTPUT_DIR, "data_drift_report.json"),
    "w"
) as file:
    json.dump(results, file, indent=4)


# Display results
print("\n========== DATA DRIFT REPORT ==========")
print(report.to_string(index=False))

if report["drift_detected"].any():
    print("\nData drift detected in some features.")
else:
    print("\nNo significant data drift detected.")

print("\nReport saved successfully!")