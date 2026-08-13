import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. LOAD DATA
# ==========================================

data_file = "data/nassau_candy_features.csv"

df = pd.read_csv(data_file)

print("Dataset loaded successfully!")


# ==========================================
# 2. SELECT FEATURES FOR CLUSTERING
# ==========================================

cluster_data = df[
    [
        "Lead Time",
        "Sales",
        "Units",
        "Gross Profit"
    ]
].copy()

print("\nClustering features:")
print(cluster_data.columns.tolist())


# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================

cluster_data = cluster_data.fillna(0)


# ==========================================
# 4. SCALE THE DATA
# ==========================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_data)


# ==========================================
# 5. CREATE CLUSTERS
# ==========================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_data)


# ==========================================
# 6. CLUSTER SUMMARY
# ==========================================

print("\n====================================")
print("CLUSTER SUMMARY")
print("====================================")

cluster_summary = df.groupby("Cluster").agg(
    Average_Lead_Time=("Lead Time", "mean"),
    Average_Sales=("Sales", "mean"),
    Average_Units=("Units", "mean"),
    Average_Profit=("Gross Profit", "mean"),
    Number_of_Orders=("Order ID", "count")
)

print(cluster_summary)


# ==========================================
# 7. IDENTIFY SLOW CLUSTER
# ==========================================

slow_cluster = cluster_summary[
    "Average_Lead_Time"
].idxmax()

print("\n====================================")
print("SLOW CLUSTER")
print("====================================")

print("Slowest Cluster:", slow_cluster)

print(
    "Average Lead Time:",
    round(
        cluster_summary.loc[
            slow_cluster,
            "Average_Lead_Time"
        ],
        2
    )
)


# ==========================================
# 8. ROUTE PERFORMANCE
# ==========================================

print("\n====================================")
print("ROUTE PERFORMANCE")
print("====================================")

route_summary = df.groupby(
    ["Region", "Ship Mode"]
).agg(
    Average_Lead_Time=("Lead Time", "mean"),
    Average_Profit=("Gross Profit", "mean"),
    Number_of_Orders=("Order ID", "count")
).sort_values(
    "Average_Lead_Time",
    ascending=False
)

print(route_summary)


# ==========================================
# 9. SLOW ROUTES
# ==========================================

print("\n====================================")
print("SLOW ROUTES")
print("====================================")

slow_routes = route_summary.head(10)

print(slow_routes)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

df.to_csv(
    "data/nassau_candy_clustered.csv",
    index=False
)

route_summary.to_csv(
    "data/route_performance.csv"
)

cluster_summary.to_csv(
    "data/cluster_summary.csv"
)

print("\n====================================")
print("CLUSTERING COMPLETE")
print("====================================")

print("Clustered data saved to:")
print("data/nassau_candy_clustered.csv")

print("Route performance saved to:")
print("data/route_performance.csv")

print("Cluster summary saved to:")
print("data/cluster_summary.csv")