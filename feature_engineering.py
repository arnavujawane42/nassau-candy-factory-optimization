import pandas as pd


# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

file_path = "data/nassau_candy_cleaned.csv"

df = pd.read_csv(file_path)

print("Cleaned dataset loaded successfully!")


# ==========================================
# 2. CONVERT DATE COLUMNS
# ==========================================

df["Order Date"] = pd.to_datetime(df["Order Date"])

df["Ship Date"] = pd.to_datetime(df["Ship Date"])


# ==========================================
# 3. CREATE TIME FEATURES
# ==========================================

df["Order Year"] = df["Order Date"].dt.year

df["Order Month"] = df["Order Date"].dt.month

df["Order Day"] = df["Order Date"].dt.day

df["Order Day of Week"] = (
    df["Order Date"].dt.dayofweek
)


# ==========================================
# 4. CREATE PROFIT MARGIN
# ==========================================

df["Profit Margin"] = (
    df["Gross Profit"] / df["Sales"]
) * 100


# ==========================================
# 5. CREATE SALES PER UNIT
# ==========================================

df["Sales Per Unit"] = (
    df["Sales"] / df["Units"]
)


# ==========================================
# 6. FACTORY MAPPING
# ==========================================

factory_mapping = {

    "Wonka Bar - Nutty Crunch Surprise":
        "Lot's O' Nuts",

    "Wonka Bar - Fudge Mallows":
        "Lot's O' Nuts",

    "Wonka Bar -Scrumdiddlyumptious":
        "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate":
        "Wicked Choccy's",

    "Wonka Bar - Triple Dazzle Caramel":
        "Wicked Choccy's",

    "Laffy Taffy":
        "Sugar Shack",

    "SweeTARTS":
        "Sugar Shack",

    "Nerds":
        "Sugar Shack",

    "Fun Dip":
        "Sugar Shack",

    "Fizzy Lifting Drinks":
        "Sugar Shack",

    "Everlasting Gobstopper":
        "Secret Factory",

    "Hair Toffee":
        "The Other Factory",

    "Lickable Wallpaper":
        "Secret Factory",

    "Wonka Gum":
        "Secret Factory",

    "Kazookles":
        "The Other Factory"
}


df["Current Factory"] = (
    df["Product Name"].map(factory_mapping)
)


# ==========================================
# 7. FACTORY COORDINATES
# ==========================================

factory_latitude = {

    "Lot's O' Nuts": 32.881893,

    "Wicked Choccy's": 32.076176,

    "Sugar Shack": 48.119140,

    "Secret Factory": 41.446333,

    "The Other Factory": 35.117500
}


# ==========================================
# 8. FACTORY LONGITUDE
# ==========================================

factory_longitude = {

    "Lot's O' Nuts": -111.768036,

    "Wicked Choccy's": -81.088371,

    "Sugar Shack": -96.181150,

    "Secret Factory": -90.565487,

    "The Other Factory": -89.971107
}

df["Factory Latitude"] = (
    df["Current Factory"].map(factory_latitude)
)

df["Factory Longitude"] = (
    df["Current Factory"].map(factory_longitude)
)


# ==========================================
# 9. REGION REPRESENTATIVE COORDINATES
# ==========================================

region_latitude = {

    "Pacific": 34.0522,

    "Atlantic": 40.7128,

    "Interior": 41.8781,

    "Gulf": 29.7604
}


region_longitude = {

    "Pacific": -118.2437,

    "Atlantic": -74.0060,

    "Interior": -87.6298,

    "Gulf": -95.3698
}


df["Region Latitude"] = (
    df["Region"].map(region_latitude)
)

df["Region Longitude"] = (
    df["Region"].map(region_longitude)
)

# ==========================================
# 10. CALCULATE FACTORY TO REGION DISTANCE
# ==========================================

import math


def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    # Convert degrees to radians

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    # Earth radius in kilometers

    earth_radius = 6371

    return earth_radius * c


df["Distance (km)"] = df.apply(
    lambda row: calculate_distance(
        row["Factory Latitude"],
        row["Factory Longitude"],
        row["Region Latitude"],
        row["Region Longitude"]
    ),
    axis=1
)

# ==========================================
# 11. CHECK FACTORY AND DISTANCE FEATURES
# ==========================================

print("\n==========================================")
print("FACTORY AND DISTANCE FEATURES")
print("==========================================")

print(
    df[
        [
            "Product Name",
            "Current Factory",
            "Region",
            "Factory Latitude",
            "Factory Longitude",
            "Region Latitude",
            "Region Longitude",
            "Distance (km)"
        ]
    ].head(10)
)

# ==========================================
# 12. SAVE FEATURE DATASET
# ==========================================

output_file = "data/nassau_candy_features.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n==========================================")
print("FEATURE ENGINEERING COMPLETE")
print("==========================================")

print("Feature dataset saved successfully!")

print(
    "File:",
    output_file
)