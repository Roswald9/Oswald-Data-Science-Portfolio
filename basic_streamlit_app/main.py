import pandas as pd            
import seaborn as sns          
import matplotlib.pyplot as plt  
import streamlit as st 

df = pd.read_csv("Data/PlayerData2526.csv")
df = df[df["POS"] != "G"]

st.title("NHL Player Salary & Performance Dashboard")
st.write(
    "This app allows users to explore NHL player performance and salary data "
    "from the 2025-26 season. Use the filters below to compare players based "
    "on age, games played, goals, assists, points, points per game, cap hit, "
    "and cost per point."
)

st.sidebar.header("Player Filters")

age_min = int(df["AGE"].min())
age_max = int(df["AGE"].max())

age_range = st.sidebar.slider(
    "Age",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

gp_min = int(df["GP"].min())
gp_max = int(df["GP"].max())

gp_range = st.sidebar.slider(
    "Games Played",
    min_value=gp_min,
    max_value=gp_max,
    value=(gp_min, gp_max)
)

caphit_min = float(df["CAPHIT"].min())
caphit_max = float(df["CAPHIT"].max())

caphit_range = st.sidebar.slider(
    "Cap Hit ($ millions)",
    min_value=caphit_min / 1_000_000,
    max_value=caphit_max / 1_000_000,
    value=(caphit_min / 1_000_000, caphit_max / 1_000_000),
    step=0.5
)

filtered_df = df[
    (df["AGE"] >= age_range[0]) &
    (df["AGE"] <= age_range[1]) &
    (df["GP"] >= gp_range[0]) &
    (df["GP"] <= gp_range[1]) &
    (df["CAPHIT"] >= caphit_range[0] * 1_000_000) &
    (df["CAPHIT"] <= caphit_range[1] * 1_000_000)
]

st.subheader("Players Found")
st.write(f"{len(filtered_df)} players match your filters.")

st.subheader("Sort Players")

sort_option = st.selectbox(
    "Sort table by:",
    [
        "Player Name",
        "Age",
        "Games Played",
        "Goals",
        "Assists",
        "Points",
        "Points Per Game",
        "Cap Hit",
        "Cost Per Point"
    ]
)

sort_order = st.selectbox(
    "Sort order:",
    ["Ascending", "Descending"]
)

ascending = sort_order == "Ascending"

if sort_option == "Player Name":
    filtered_df = filtered_df.sort_values("PLAYERS", ascending=ascending)
elif sort_option == "Age":
    filtered_df = filtered_df.sort_values("AGE", ascending=ascending)
elif sort_option == "Games Played":
    filtered_df = filtered_df.sort_values("GP", ascending=ascending)
elif sort_option == "Goals":
    filtered_df = filtered_df.sort_values("G", ascending=ascending)
elif sort_option == "Points Per Game":
    filtered_df = filtered_df.sort_values("PPG", ascending=ascending)
elif sort_option == "Cap Hit":
    filtered_df = filtered_df.sort_values("CAPHIT", ascending=ascending)
elif sort_option == "Cost Per Point":
    filtered_df = filtered_df.sort_values("CPP", ascending=ascending)
elif sort_option == "Assists":
    filtered_df = filtered_df.sort_values("A", ascending=ascending)
elif sort_option == "Points":
    filtered_df = filtered_df.sort_values("PTS", ascending=ascending)

st.dataframe(
    filtered_df[
        ["PLAYERS", "AGE", "GP", "G", "A", "PTS", "PPG", "CAPHIT", "CPP"]
    ],
    hide_index=True
)