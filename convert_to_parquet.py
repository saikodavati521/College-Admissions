"""Convert train_admissions_dataset.csv to parquet format."""

import pandas as pd
from pathlib import Path

# Define paths
script_dir = Path(__file__).parent
csv_path = script_dir / "model_training" / "train_data" / "train_admissions_dataset.csv"
parquet_path = script_dir / "model_training" / "train_data" / "train_admissions_dataset.parquet"

# Read CSV file
print(f"Reading CSV file: {csv_path}")
df = pd.read_csv(csv_path)

# Save as parquet
print(f"Converting to parquet: {parquet_path}")
df.to_parquet(parquet_path, index=False)

print(f"✓ Parquet file created successfully!")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  Location: {parquet_path}")
