import pandas as pd
import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

data_file = "data/nassau_candy_features.csv"

df = pd.read_csv(data_file)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "best_lead_time_model.pkl"
)

print("Model loaded successfully!")


# ==========================================
# 3. FACTORY LIST
# ==========================================

factories = [

    "Lot's O' Nuts",

    "Wicked Choccy's",

    "Sugar Shack",

    "Secret Factory",

    "The Other Factory"
]


# ==========================================
# 4. SELECT PRODUCT
# ==========================================

product = input(
    "\nEnter Product Name: "
)


# Check product

if product not in df["Product Name"].unique():

    print("\nProduct not found.")

    print("\nAvailable products:")

    for item in df["Product Name"].unique():

        print("-", item)

    exit()


# ==========================================
# 5. SELECT REGION
# ==========================================

print("\nAvailable Regions:")

regions = df["Region"].unique()

for region in regions:

    print("-", region)


region = input("\nEnter Region: ").strip().title()


if region not in regions:

    print("Invalid region.")

    exit()


# ==========================================
# 6. SELECT SHIP MODE
# ==========================================

print("\nAvailable Ship Modes:")

ship_modes = df["Ship Mode"].unique()

for mode in ship_modes:

    print("-", mode)


ship_mode = input(
    "\nEnter Ship Mode: "
)


if ship_mode not in ship_modes:

    print("Invalid ship mode.")

    exit()


# ==========================================
# 7. GET PRODUCT DATA
# ==========================================

product_data = df[
    df["Product Name"] == product
]


# ==========================================
# 8. CURRENT FACTORY
# ==========================================

current_factory = (
    product_data["Current Factory"]
    .iloc[0]
)


division = (
    product_data["Division"]
    .iloc[0]
)


# ==========================================
# 9. GET TYPICAL NUMERICAL VALUES
# ==========================================

filtered_data = df[
    (df["Product Name"] == product) &
    (df["Region"] == region) &
    (df["Ship Mode"] == ship_mode)
]


# If exact combination doesn't exist,
# use product-level data

if len(filtered_data) == 0:

    filtered_data = product_data


median_units = (
    filtered_data["Units"].median()
)

median_sales = (
    filtered_data["Sales"].median()
)

median_cost = (
    filtered_data["Cost"].median()
)

median_profit = (
    filtered_data["Gross Profit"].median()
)

median_year = (
    filtered_data["Order Year"].median()
)

median_month = (
    filtered_data["Order Month"].median()
)

median_day = (
    filtered_data["Order Day"].median()
)

median_day_week = (
    filtered_data["Order Day of Week"].median()
)


# ==========================================
# 10. CREATE SCENARIOS
# ==========================================

scenarios = []


for factory in factories:

    scenario = {

        "Product Name": product,

        "Current Factory": factory,

        "Region": region,

        "Ship Mode": ship_mode,

        "Division": division,

        "Units": median_units,

        "Sales": median_sales,

        "Cost": median_cost,

        "Gross Profit": median_profit,

        "Order Year": median_year,

        "Order Month": median_month,

        "Order Day": median_day,

        "Order Day of Week": median_day_week
    }


    scenarios.append(
        scenario
    )


scenario_df = pd.DataFrame(
    scenarios
)


# ==========================================
# 11. PREDICT LEAD TIMES
# ==========================================

scenario_df["Predicted Lead Time"] = (
    model.predict(scenario_df)
)


# ==========================================
# 12. CURRENT FACTORY PREDICTION
# ==========================================

current_prediction = (
    scenario_df[
        scenario_df["Current Factory"]
        == current_factory
    ]["Predicted Lead Time"]
    .iloc[0]
)

# ==========================================
# 13. CALCULATE IMPROVEMENT
# ==========================================

scenario_df["Lead Time Reduction"] = (
    current_prediction
    - scenario_df["Predicted Lead Time"]
)

scenario_df["Lead Time Reduction %"] = (
    scenario_df["Lead Time Reduction"]
    / current_prediction
) * 100


# ==========================================
# 14. PROFIT SENSITIVITY
# ==========================================

scenario_df["Profit Margin %"] = (
    scenario_df["Gross Profit"]
    / scenario_df["Sales"]
) * 100

# Factory-specific cost data is not available
# in the current dataset.
# Therefore profit is treated as stable.

scenario_df["Profit Impact"] = 0

scenario_df["Profit Impact %"] = 0


# ==========================================
# 15. CONFIDENCE AND RISK
# ==========================================

sample_count = len(filtered_data)

scenario_df["Sample Count"] = sample_count

confidence_score = min(
    (sample_count / 100) * 100,
    100
)

scenario_df["Confidence Score"] = confidence_score


if confidence_score >= 75:
    risk_level = "Low"

elif confidence_score >= 40:
    risk_level = "Medium"

else:
    risk_level = "High"

scenario_df["Risk Level"] = risk_level

# ==========================================
# 16. RANK FACTORIES
# ==========================================

scenario_df = scenario_df.sort_values(

    "Predicted Lead Time"

)


# ==========================================
# 17. DISPLAY RESULTS
# ==========================================

print("\n==========================================")

print("FACTORY OPTIMIZATION RESULTS")

print("==========================================")


print("\nProduct:",
      product)

print("Region:",
      region)

print("Ship Mode:",
      ship_mode)

print("Current Factory:",
      current_factory)


print(
    "\nCurrent Predicted Lead Time:",
    round(current_prediction, 2),
    "days"
)


print("\nFactory Scenarios:\n")


for _, row in scenario_df.iterrows():

    print(
        row["Current Factory"],
        "→",
        round(
            row["Predicted Lead Time"],
            2
        ),
        "days | Reduction:",
        round(
            row["Lead Time Reduction"],
            2
        ),
        "days |",
        round(
            row["Lead Time Reduction %"],
            2
        ),
        "% | Risk:",
        row["Risk Level"]
    )


# ==========================================
# 18. BEST ALTERNATIVE
# ==========================================

# ==========================================
# 18. RECOMMENDATION LOGIC
# ==========================================

alternatives = scenario_df[
    scenario_df["Current Factory"] != current_factory
].copy()


# Find the alternative with the lowest
# predicted lead time

best_alternative = alternatives.iloc[0]


print("\n==========================================")
print("RECOMMENDATION")
print("==========================================")


# Check whether alternative is actually better

if best_alternative["Predicted Lead Time"] < current_prediction:

    print(
        "Recommended Factory:",
        best_alternative["Current Factory"]
    )

    print(
        "Predicted Lead Time:",
        round(
            best_alternative["Predicted Lead Time"],
            2
        ),
        "days"
    )

    print(
        "Lead Time Reduction:",
        round(
            best_alternative["Lead Time Reduction"],
            2
        ),
        "days"
    )

    print(
        "Lead Time Reduction:",
        round(
            best_alternative["Lead Time Reduction %"],
            2
        ),
        "%"
    )

    print(
    "Confidence Score:",
    round(
        best_alternative["Confidence Score"],
        2
    ),
    "%"
)

    print(
    "Risk Level:",
    best_alternative["Risk Level"]
)

    print(
    "Profit Impact: Stable"
)

else:

    print(
        "Recommendation: KEEP CURRENT FACTORY"
    )

    print(
        "Current Factory:",
        current_factory
    )

    print(
        "Current Predicted Lead Time:",
        round(
            current_prediction,
            2
        ),
        "days"
    )

print(
    "Profit Margin:",
    round(
        scenario_df["Profit Margin %"].iloc[0],
        2
    ),
    "%"
)

print(
    "Profit Impact: Stable"
)

print(
    "Confidence Score:",
    round(confidence_score, 2),
    "%"
)

print(
    "Risk Level:",
    risk_level
)

print(
        "Reason: No alternative factory provides"
        " a lower predicted lead time."
    )


# ==========================================
# 19. SAVE RESULTS
# ==========================================

scenario_df.to_csv(

    "data/scenario_results.csv",

    index=False
)


print(
    "\nScenario results saved to:",
    "data/scenario_results.csv"
)