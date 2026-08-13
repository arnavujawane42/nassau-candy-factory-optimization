import pandas as pd
import subprocess
import sys

# ==========================================
# NASSAU CANDY OPTIMIZATION
# MAIN PROJECT PIPELINE
# ==========================================

print("\n==========================================")
print("   NASSAU CANDY OPTIMIZATION PROJECT")
print("==========================================")

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "data/Nassau_Candy_Data.xlsx"

df = pd.read_excel(file_path, engine="openpyxl")

print("\nDataset loaded successfully!")

# ==========================================
# 2. CONVERT DATE COLUMNS
# ==========================================

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# ==========================================
# 3. CREATE LEAD TIME
# ==========================================

df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\n==========================================")
print("MISSING VALUES")
print("==========================================")

print(df.isnull().sum())

# ==========================================
# 5. CHECK DUPLICATES
# ==========================================

print("\n==========================================")
print("DUPLICATE RECORDS")
print("==========================================")

print("Duplicate rows:", df.duplicated().sum())

# ==========================================
# 6. DATASET INFORMATION
# ==========================================

print("\n==========================================")
print("DATASET INFORMATION")
print("==========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("Unique Products:", df["Product Name"].nunique())
print("Unique Customers:", df["Customer ID"].nunique())
print("Unique Regions:", df["Region"].nunique())
print("Unique Ship Modes:", df["Ship Mode"].nunique())

# ==========================================
# 7. BUSINESS SUMMARY
# ==========================================

print("\n==========================================")
print("BUSINESS SUMMARY")
print("==========================================")

total_sales = df["Sales"].sum()
total_profit = df["Gross Profit"].sum()
total_cost = df["Cost"].sum()
total_units = df["Units"].sum()

print("Total Sales:", round(total_sales, 2))
print("Total Gross Profit:", round(total_profit, 2))
print("Total Cost:", round(total_cost, 2))
print("Total Units:", total_units)

# ==========================================
# 8. GROSS MARGIN
# ==========================================

gross_margin = (
    total_profit / total_sales
) * 100

print("Gross Margin:", round(gross_margin, 2), "%")

# ==========================================
# 9. LEAD TIME SUMMARY
# ==========================================

print("\n==========================================")
print("LEAD TIME SUMMARY")
print("==========================================")

print(df["Lead Time"].describe())

# ==========================================
# 10. SALES BY REGION
# ==========================================

print("\n==========================================")
print("SALES BY REGION")
print("==========================================")

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(region_sales)

# ==========================================
# 11. PROFIT BY REGION
# ==========================================

print("\n==========================================")
print("PROFIT BY REGION")
print("==========================================")

region_profit = (
    df.groupby("Region")["Gross Profit"]
    .sum()
    .sort_values(ascending=False)
)

print(region_profit)

# ==========================================
# 12. TOP PRODUCTS BY SALES
# ==========================================

print("\n==========================================")
print("TOP PRODUCTS BY SALES")
print("==========================================")

product_sales = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(product_sales)

# ==========================================
# 13. TOP PRODUCTS BY PROFIT
# ==========================================

print("\n==========================================")
print("TOP PRODUCTS BY PROFIT")
print("==========================================")

product_profit = (
    df.groupby("Product Name")["Gross Profit"]
    .sum()
    .sort_values(ascending=False)
)

print(product_profit)

# ==========================================
# 14. SAVE CLEANED DATA
# ==========================================

df.to_csv(
    "data/nassau_candy_cleaned.csv",
    index=False
)

print("\n==========================================")
print("DATA CLEANING COMPLETE")
print("==========================================")

print("Cleaned dataset saved successfully!")

# ==========================================
# 15. RUN FEATURE ENGINEERING
# ==========================================

print("\n==========================================")
print("RUNNING FEATURE ENGINEERING")
print("==========================================")

result = subprocess.run(
    [sys.executable, "feature_engineering.py"]
)

if result.returncode != 0:
    print("\nFeature Engineering failed.")
    sys.exit(1)

print("\nFeature Engineering completed successfully!")

# ==========================================
# 16. RUN MACHINE LEARNING
# ==========================================

print("\n==========================================")
print("RUNNING MACHINE LEARNING")
print("==========================================")

result = subprocess.run(
    [sys.executable, "model.py"]
)

if result.returncode != 0:
    print("\nMachine Learning failed.")
    sys.exit(1)

print("\nMachine Learning completed successfully!")

# ==========================================
# 17. RUN SCENARIO ENGINE
# ==========================================

print("\n==========================================")
print("RUNNING SCENARIO ENGINE")
print("==========================================")

result = subprocess.run(
    [sys.executable, "scenario_engine.py"]
)

if result.returncode != 0:
    print("\nScenario Engine failed.")
    sys.exit(1)

print("\nScenario Engine completed successfully!")

# ==========================================
# 18. PROJECT COMPLETE
# ==========================================

print("\n==========================================")
print("   NASSAU CANDY OPTIMIZATION COMPLETE")
print("==========================================")

print("\nAll project stages completed successfully!")

print("\nGenerated Files:")
print("- data/nassau_candy_cleaned.csv")
print("- data/nassau_candy_features.csv")
print("- data/model_results.csv")
print("- best_lead_time_model.pkl")
print("- data/scenario_results.csv")

print("\n==========================================")
print("FINAL PIPELINE FINISHED")
print("==========================================")