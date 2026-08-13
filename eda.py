import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

file_path = "data/nassau_candy_cleaned.csv"

df = pd.read_csv(file_path)

print("Cleaned dataset loaded successfully!")


# ==========================================
# 2. CREATE CHARTS
# ==========================================


# ------------------------------------------
# Chart 1: Sales by Region
# ------------------------------------------

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

region_sales.plot(kind="bar")

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("charts/sales_by_region.png")

plt.show()


# ------------------------------------------
# Chart 2: Profit by Region
# ------------------------------------------

region_profit = (
    df.groupby("Region")["Gross Profit"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

region_profit.plot(kind="bar")

plt.title("Total Gross Profit by Region")
plt.xlabel("Region")
plt.ylabel("Gross Profit")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("charts/profit_by_region.png")

plt.show()


# ------------------------------------------
# Chart 3: Sales by Product
# ------------------------------------------

product_sales = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

product_sales.plot(kind="bar")

plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")

plt.xticks(rotation=75, ha="right")

plt.tight_layout()

plt.savefig("charts/sales_by_product.png")

plt.show()


# ------------------------------------------
# Chart 4: Profit by Product
# ------------------------------------------

product_profit = (
    df.groupby("Product Name")["Gross Profit"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

product_profit.plot(kind="bar")

plt.title("Gross Profit by Product")
plt.xlabel("Product")
plt.ylabel("Gross Profit")

plt.xticks(rotation=75, ha="right")

plt.tight_layout()

plt.savefig("charts/profit_by_product.png")

plt.show()


# ------------------------------------------
# Chart 5: Lead Time by Ship Mode
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Ship Mode",
    y="Lead Time"
)

plt.title("Lead Time Distribution by Ship Mode")
plt.xlabel("Ship Mode")
plt.ylabel("Lead Time (Days)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("charts/lead_time_by_ship_mode.png")

plt.show()


# ------------------------------------------
# Chart 6: Sales by Division
# ------------------------------------------

division_sales = (
    df.groupby("Division")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

division_sales.plot(kind="bar")

plt.title("Total Sales by Division")
plt.xlabel("Division")
plt.ylabel("Sales")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("charts/sales_by_division.png")

plt.show()


# ------------------------------------------
# Chart 7: Profit by Division
# ------------------------------------------

division_profit = (
    df.groupby("Division")["Gross Profit"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

division_profit.plot(kind="bar")

plt.title("Total Gross Profit by Division")
plt.xlabel("Division")
plt.ylabel("Gross Profit")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("charts/profit_by_division.png")

plt.show()


print("\n======================================")
print("EDA COMPLETE")
print("======================================")

print("All charts saved inside the charts folder.")