import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Nassau Candy Optimization",
    page_icon="🍫",
    layout="wide"
)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv("nassau_candy_features.csv")

model = joblib.load("best_lead_time_model.pkl")


# ============================================================
# FACTORIES
# ============================================================

factories = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory"
]


# ============================================================
# TITLE
# ============================================================

st.title("🍫 Nassau Candy Factory Optimization")

st.write(
    "Factory Reallocation & Shipping Optimization "
    "Recommendation System"
)

st.caption(
    "Predict shipping performance, compare factory scenarios, "
    "and identify the most suitable factory assignment."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Scenario Selection")


product = st.sidebar.selectbox(
    "Select Product",
    sorted(df["Product Name"].dropna().unique())
)


region = st.sidebar.selectbox(
    "Select Region",
    sorted(df["Region"].dropna().unique())
)


ship_mode = st.sidebar.selectbox(
    "Select Ship Mode",
    sorted(df["Ship Mode"].dropna().unique())
)


st.sidebar.subheader("Optimization Priority")

priority = st.sidebar.slider(
    "Speed vs Profit",
    min_value=0,
    max_value=100,
    value=50,
    step=10
)


speed_weight = priority / 100
profit_weight = (100 - priority) / 100


st.sidebar.write(
    f"Speed Priority: **{priority}%**"
)

st.sidebar.write(
    f"Profit Priority: **{100 - priority}%**"
)


# ============================================================
# PRODUCT INFORMATION
# ============================================================

product_data = df[
    df["Product Name"] == product
].copy()


current_factory = (
    product_data["Current Factory"].iloc[0]
)


division = (
    product_data["Division"].iloc[0]
)


# ============================================================
# SELECTED PRODUCT
# ============================================================

st.divider()

st.subheader("Selected Product")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Product",
        product
    )

with col2:
    st.metric(
        "Current Factory",
        current_factory
    )

with col3:
    st.metric(
        "Division",
        division
    )


# ============================================================
# FILTER SCENARIO DATA
# ============================================================

filtered_data = df[
    (df["Product Name"] == product) &
    (df["Region"] == region) &
    (df["Ship Mode"] == ship_mode)
].copy()


# If exact combination does not exist,
# use product-level historical data.

used_fallback = False

if len(filtered_data) == 0:

    filtered_data = product_data.copy()

    used_fallback = True


sample_count = len(filtered_data)


# ============================================================
# MEDIAN VALUES
# ============================================================

median_units = filtered_data["Units"].median()

median_sales = filtered_data["Sales"].median()

median_cost = filtered_data["Cost"].median()

median_profit = filtered_data["Gross Profit"].median()

median_year = filtered_data["Order Year"].median()

median_month = filtered_data["Order Month"].median()

median_day = filtered_data["Order Day"].median()

median_day_week = filtered_data["Order Day of Week"].median()


# ============================================================
# CREATE FACTORY SCENARIOS
# ============================================================

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

    scenarios.append(scenario)


scenario_df = pd.DataFrame(scenarios)


# ============================================================
# PREDICT LEAD TIME
# ============================================================

scenario_df["Predicted Lead Time"] = model.predict(
    scenario_df
)


# ============================================================
# CURRENT FACTORY PREDICTION
# ============================================================

current_prediction = scenario_df.loc[
    scenario_df["Current Factory"] == current_factory,
    "Predicted Lead Time"
].iloc[0]


# ============================================================
# LEAD TIME IMPROVEMENT
# ============================================================

scenario_df["Lead Time Reduction"] = (
    current_prediction
    - scenario_df["Predicted Lead Time"]
)


if current_prediction != 0:

    scenario_df["Lead Time Reduction %"] = (
        scenario_df["Lead Time Reduction"]
        / current_prediction
    ) * 100

else:

    scenario_df["Lead Time Reduction %"] = 0


# ============================================================
# PROFIT MARGIN
# ============================================================

if median_sales != 0:

    profit_margin = (
        median_profit
        / median_sales
    ) * 100

else:

    profit_margin = 0


scenario_df["Profit Margin %"] = profit_margin


# ============================================================
# CONFIDENCE SCORE
# ============================================================

confidence_score = min(
    sample_count,
    100
)


# ============================================================
# RISK LEVEL
# ============================================================

if confidence_score >= 75:

    risk_level = "Low"

elif confidence_score >= 40:

    risk_level = "Medium"

else:

    risk_level = "High"


scenario_df["Confidence Score"] = confidence_score

scenario_df["Risk Level"] = risk_level


# ============================================================
# FACTORY OPTIMIZATION SIMULATOR
# ============================================================

st.divider()

st.subheader("Factory Optimization Simulator")

st.write(
    "Predicted shipping performance for the selected "
    "product across all available factories."
)


display_df = scenario_df[
    [
        "Current Factory",
        "Predicted Lead Time",
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Risk Level",
        "Confidence Score"
    ]
].copy()


display_df.columns = [
    "Factory",
    "Predicted Lead Time (Days)",
    "Lead Time Reduction (Days)",
    "Lead Time Reduction (%)",
    "Risk Level",
    "Confidence Score (%)"
]


display_df[
    "Predicted Lead Time (Days)"
] = display_df[
    "Predicted Lead Time (Days)"
].round(2)


display_df[
    "Lead Time Reduction (Days)"
] = display_df[
    "Lead Time Reduction (Days)"
].round(2)


display_df[
    "Lead Time Reduction (%)"
] = display_df[
    "Lead Time Reduction (%)"
].round(2)


display_df[
    "Confidence Score (%)"
] = display_df[
    "Confidence Score (%)"
].round(2)


st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# OPTIMIZATION SCORING
# ============================================================

st.divider()

st.subheader("Recommendation Dashboard")


# ------------------------------------------------------------
# SPEED SCORE
# ------------------------------------------------------------

max_lead_time = scenario_df[
    "Predicted Lead Time"
].max()


min_lead_time = scenario_df[
    "Predicted Lead Time"
].min()


if max_lead_time != min_lead_time:

    scenario_df["Speed Score"] = (
        (
            max_lead_time
            - scenario_df["Predicted Lead Time"]
        )
        /
        (
            max_lead_time
            - min_lead_time
        )
    ) * 100

else:

    scenario_df["Speed Score"] = 100


# ------------------------------------------------------------
# PROFIT PRESERVATION SCORE
# ------------------------------------------------------------

# The dataset does not contain factory-specific
# manufacturing costs for hypothetical reassignment.
#
# Therefore we use the observed product profit margin
# as a profit-preservation indicator.

scenario_df["Profit Preservation Score"] = (
    scenario_df["Gross Profit"]
    /
    scenario_df["Sales"].replace(0, pd.NA)
) * 100


scenario_df["Profit Preservation Score"] = (
    scenario_df["Profit Preservation Score"]
    .fillna(0)
    .clip(0, 100)
)


# ------------------------------------------------------------
# COMPOSITE SCORE
# ------------------------------------------------------------

scenario_df["Optimization Score"] = (

    scenario_df["Speed Score"]
    * speed_weight

    +

    scenario_df["Profit Preservation Score"]
    * profit_weight

)


# ============================================================
# RANK FACTORIES
# ============================================================

ranked_scenarios = scenario_df.sort_values(
    "Optimization Score",
    ascending=False
).reset_index(drop=True)


ranked_scenarios["Rank"] = (
    ranked_scenarios.index + 1
)


# ============================================================
# BEST FACTORY
# ============================================================

best_factory = ranked_scenarios.iloc[0]


# ============================================================
# PRIORITY DISPLAY
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Speed Priority",
        f"{priority}%"
    )


with col2:

    st.metric(
        "Profit Priority",
        f"{100 - priority}%"
    )


st.info(
    "The recommendation score combines predicted shipping "
    "speed with the observed profit margin according to "
    "the selected priority."
)


# ============================================================
# RANKED RECOMMENDATIONS
# ============================================================

st.write("### Ranked Factory Recommendations")


recommendation_df = ranked_scenarios[
    [
        "Rank",
        "Current Factory",
        "Predicted Lead Time",
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Speed Score",
        "Profit Preservation Score",
        "Optimization Score",
        "Risk Level"
    ]
].copy()


recommendation_df.columns = [
    "Rank",
    "Factory",
    "Predicted Lead Time (Days)",
    "Lead Time Reduction (Days)",
    "Lead Time Reduction (%)",
    "Speed Score",
    "Profit Score",
    "Optimization Score",
    "Risk Level"
]


for column in [
    "Predicted Lead Time (Days)",
    "Lead Time Reduction (Days)",
    "Lead Time Reduction (%)",
    "Speed Score",
    "Profit Score",
    "Optimization Score"
]:

    recommendation_df[column] = (
        recommendation_df[column].round(2)
    )


st.dataframe(
    recommendation_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

st.write("### Final Recommendation")


if best_factory["Current Factory"] == current_factory:

    st.success(
        "KEEP CURRENT FACTORY"
    )

    recommendation_text = (
        f"The current factory, {current_factory}, "
        "remains the best option under the selected "
        "optimization priorities."
    )

else:

    st.success(
        f"RECOMMENDED FACTORY: "
        f"{best_factory['Current Factory']}"
    )

    recommendation_text = (
        f"Reallocate {product} from "
        f"{current_factory} to "
        f"{best_factory['Current Factory']} "
        f"if the organization accepts the scenario."
    )


st.write(recommendation_text)


# ============================================================
# RECOMMENDATION KPIs
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current Factory",
        current_factory
    )


with col2:

    st.metric(
        "Recommended Factory",
        best_factory["Current Factory"]
    )


with col3:

    st.metric(
        "Predicted Lead Time",
        f"{best_factory['Predicted Lead Time']:.2f} days"
    )


with col4:

    st.metric(
        "Optimization Score",
        f"{best_factory['Optimization Score']:.2f}"
    )


# ============================================================
# WHAT-IF SCENARIO ANALYSIS
# ============================================================

st.divider()

st.subheader("What-If Scenario Analysis")

st.write(
    "Compare the current factory with the "
    "recommended factory."
)


comparison = scenario_df[
    scenario_df["Current Factory"].isin(
        [
            current_factory,
            best_factory["Current Factory"]
        ]
    )
].copy()


comparison_display = comparison[
    [
        "Current Factory",
        "Predicted Lead Time",
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Optimization Score",
        "Risk Level"
    ]
].copy()


comparison_display.columns = [
    "Factory",
    "Predicted Lead Time (Days)",
    "Lead Time Reduction (Days)",
    "Lead Time Reduction (%)",
    "Optimization Score",
    "Risk Level"
]


for column in [
    "Predicted Lead Time (Days)",
    "Lead Time Reduction (Days)",
    "Lead Time Reduction (%)",
    "Optimization Score"
]:

    comparison_display[column] = (
        comparison_display[column].round(2)
    )


st.dataframe(
    comparison_display,
    width="stretch",
    hide_index=True
)


# ============================================================
# LEAD TIME CHART
# ============================================================

st.write("### Lead Time Comparison")


chart_data = comparison.set_index(
    "Current Factory"
)[
    "Predicted Lead Time"
]


st.bar_chart(chart_data)


# ============================================================
# IMPROVEMENT MESSAGE
# ============================================================

if best_factory["Current Factory"] != current_factory:

    improvement_days = (
        current_prediction
        - best_factory["Predicted Lead Time"]
    )


    if current_prediction != 0:

        improvement_percent = (
            improvement_days
            / current_prediction
        ) * 100

    else:

        improvement_percent = 0


    st.success(
        f"Moving from {current_factory} to "
        f"{best_factory['Current Factory']} is predicted "
        f"to reduce lead time by "
        f"{improvement_days:.2f} days "
        f"({improvement_percent:.2f}%)."
    )

else:

    st.info(
        "The current factory remains the best option "
        "under the selected optimization priorities."
    )


# ============================================================
# RISK & IMPACT PANEL
# ============================================================

st.divider()

st.subheader("Risk & Impact Panel")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Confidence Score",
        f"{confidence_score:.1f}%"
    )


with col2:

    st.metric(
        "Risk Level",
        risk_level
    )


with col3:

    st.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )


with col4:

    st.metric(
        "Sample Count",
        sample_count
    )


# ============================================================
# RISK MESSAGE
# ============================================================

if risk_level == "Low":

    st.success(
        "Low Risk: The selected scenario has sufficient "
        "historical observations."
    )

elif risk_level == "Medium":

    st.warning(
        "Medium Risk: The scenario has a moderate "
        "number of historical observations."
    )

else:

    st.error(
        "High Risk: Limited historical data is available "
        "for this scenario."
    )


# ============================================================
# FALLBACK NOTICE
# ============================================================

if used_fallback:

    st.warning(
        "The selected Product + Region + Ship Mode "
        "combination was not found exactly in the historical "
        "data. Product-level historical values were used "
        "for the scenario simulation."
    )


# ============================================================
# PROFIT IMPACT
# ============================================================

st.write("### Profit Impact")


st.info(
    "Direct factory-level profit impact cannot be calculated "
    "from the current dataset because factory-specific "
    "manufacturing costs for hypothetical reassignment "
    "are not available. The observed product profit margin "
    "is therefore used as a profit-preservation indicator."
)


st.write(
    f"Current estimated profit margin: "
    f"**{profit_margin:.2f}%**"
)


# ============================================================
# DECISION SUMMARY
# ============================================================

st.write("### Decision Summary")


if best_factory["Current Factory"] != current_factory:

    st.success(
        f"Recommended action: consider reallocating "
        f"{product} from {current_factory} to "
        f"{best_factory['Current Factory']}. "
        f"The model predicts a lead-time improvement of "
        f"{best_factory['Lead Time Reduction']:.2f} days."
    )

else:

    st.info(
        "Recommended action: KEEP CURRENT FACTORY. "
        "No alternative factory achieves a better "
        "overall optimization score under the selected "
        "speed-versus-profit priorities."
    )


# ============================================================
# SAVE CURRENT SCENARIO RESULTS
# ============================================================

scenario_df.to_csv(
    "scenario_results.csv",
    index=False
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Nassau Candy Factory Optimization | "
    "Predictive Analytics & Decision Support System"
)