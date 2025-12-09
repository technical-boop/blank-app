# Write this whole script to the given path
"""
Streamlit app for Bhutan Under-5 Mortality Forecasting

This app:
- Loads cleaned WHO data for Bhutan
- Focuses on one indicator: "Number of deaths in children aged <5 years, by cause"
- Aggregates total deaths per year
- Loads a pre-trained LinearRegression model (under5_linear.pkl)
- Lets user select a year and see predicted total under-5 deaths
"""

import streamlit as st            # Streamlit for web UI
import pandas as pd              # Pandas for data handling
import joblib                    # joblib to load the trained model
import matplotlib.pyplot as plt  # Matplotlib for plots
from pathlib import Path         # Path from pathlib to build file paths that work anywhere

# -----------------------------
# 0. Base paths (work in Colab + Streamlit Cloud)
# -----------------------------

# __file__ = path of this script (app/streamlit_app.py) when run by Streamlit
BASE_DIR = Path(__file__).resolve().parent.parent   # Go up one level → project root

# Build paths relative to project root (no /content/ hardcoding)
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_health.csv"  # data/processed/cleaned_health.csv
MODEL_PATH = BASE_DIR / "models" / "under5_linear.pkl"              # models/under5_linear.pkl


# -----------------------------
# 1. Load data and model
# -----------------------------

@st.cache_data  # Cache data so it doesn't reload on every interaction
def load_data():
    """Load the cleaned WHO Bhutan dataset and prepare yearly under-5 deaths."""
    df = pd.read_csv(DATA_PATH)  # Read CSV from our relative path

    # Focus only on our chosen indicator
    indicator = "Number of deaths in children aged <5 years, by cause"  # Indicator of interest
    df_indicator = df[df["indicator_name"] == indicator].copy()         # Filter rows for that indicator

    # Aggregate: sum across all causes for each year
    df_yearly = df_indicator.groupby("year")["value"].sum().reset_index()  # Group by year and sum deaths

    # Ensure year is numeric and sorted
    df_yearly["year"] = pd.to_numeric(df_yearly["year"], errors="coerce")  # Convert year to numeric
    df_yearly = df_yearly.dropna(subset=["year", "value"])                 # Drop rows with missing year/value
    df_yearly = df_yearly.sort_values("year")                              # Sort by year ascending

    return df_yearly  # Cleaned, aggregated yearly dataset


@st.cache_resource  # Cache model so it's loaded only once
def load_model():
    """Load the trained Linear Regression model for under-5 deaths."""
    model = joblib.load(MODEL_PATH)  # Load model from models/under5_linear.pkl
    return model


# Load data and model once at the top-level
df_yearly = load_data()   # Historical under-5 deaths by year
model = load_model()      # LinearRegression model

# Pre-compute min and max year for scaling and UI
min_year = int(df_yearly["year"].min())  # Earliest year in dataset
max_year = int(df_yearly["year"].max())  # Latest year in dataset


# -----------------------------
# 2. Streamlit page layout
# -----------------------------

st.title("Bhutan Under-5 Mortality Forecasting Dashboard")  # Main title of the app
st.write(
    """
This dashboard uses WHO data for Bhutan to analyze and forecast
the **total number of deaths in children under 5 years of age**
(across all causes combined).

The model is a simple Linear Regression trained on historical data
from around 2000–2021, so forecasts far into the future should be
treated as *illustrative only*, not as official statistics.
"""
)

# Show raw data preview (first few rows)
st.subheader("1. Historical Under-5 Deaths (Aggregated by Year)")
st.dataframe(df_yearly.head())  # Display small preview of the dataframe


# -----------------------------
# 3. Historical trend plot
# -----------------------------

st.subheader("2. Trend of Total Under-5 Deaths in Bhutan")

fig, ax = plt.subplots(figsize=(8, 4))                      # Create a figure and axis
ax.plot(df_yearly["year"], df_yearly["value"], marker="o")  # Line plot of deaths over year
ax.set_title("Total Under-5 Deaths in Bhutan (All Causes Combined)")  # Plot title
ax.set_xlabel("Year")                                       # X-axis label
ax.set_ylabel("Number of Deaths")                           # Y-axis label
ax.grid(True)                                               # Add grid for readability

st.pyplot(fig)  # Render the plot inside Streamlit


# -----------------------------
# 4. Forecasting widget
# -----------------------------

st.subheader("3. Forecast Future Under-5 Deaths")

st.write(
    f"""
The model was trained on data from **{min_year}** to **{max_year}**.
You can choose a year within this range to see the actual value,
or choose a future year to see the model's prediction.
"""
)

# Year selection slider (allow some future years beyond training data)
selected_year = st.slider(
    "Select a year to forecast:",   # Label shown in UI
    min_value=min_year,             # Minimum selectable year
    max_value=max_year + 20,        # Allow up to ~20 years beyond last observed year
    value=max_year + 5,             # Default selected year on load
    step=1                          # Step by 1 year
)


# Function to compute model prediction for any year
def predict_deaths(year: int):
    """Predict total under-5 deaths for a given year, clipped at zero."""
    # Scale the year same way as in training:
    # year_scaled = (year - min_year) / (max_year - min_year)
    year_scaled = (year - min_year) / (max_year - min_year)

    # Create a DataFrame with correct feature name for the model
    X_future = pd.DataFrame({"year_scaled": [year_scaled]})  # Shape (1, 1)

    # Raw model prediction (could be negative in extreme cases)
    raw_pred = model.predict(X_future)[0]

    # Clip at 0 because number of deaths cannot be negative
    clipped_pred = max(raw_pred, 0)

    return raw_pred, clipped_pred  # Return both raw and clipped values


# Check if selected year is in historical data
historical_row = df_yearly[df_yearly["year"] == selected_year]  # Filter df for the chosen year

if not historical_row.empty:
    # If year exists in historical data, show actual value instead of prediction
    actual_value = float(historical_row["value"].values[0])  # Extract actual deaths
    st.markdown(
        f"### 📘 Year {selected_year}: Actual total under-5 deaths = **{int(actual_value)}**"
    )
else:
    # If future (or missing) year, use the model for prediction
    raw_pred, clipped_pred = predict_deaths(selected_year)  # Get prediction
    st.markdown(
        f"### 🔮 Year {selected_year}: Predicted total under-5 deaths = **{clipped_pred:,.2f}**"
    )
    st.caption(
        f"(Raw model output: {raw_pred:,.2f}. "
        "Values below 0 are clipped to 0 because deaths cannot be negative.)"
    )

# -----------------------------
# 5. Plot with forecast point
# -----------------------------

st.subheader("4. Historical Data with Forecast Point")

# Prepare base historical data as lists
years = df_yearly["year"].tolist()
values = df_yearly["value"].tolist()

# If forecasting a future year, append forecast to the line
forecast_added = False
if selected_year > max_year:
    _, clipped_pred = predict_deaths(selected_year)
    years.append(selected_year)
    values.append(clipped_pred)
    forecast_added = True

# Create the figure
fig2, ax2 = plt.subplots(figsize=(8, 4))

# Plot the (possibly extended) line
ax2.plot(years, values, marker="o", label="Historical + Forecast")

# Highlight forecast point if it exists
if forecast_added:
    ax2.scatter([selected_year], [clipped_pred], s=80, label="Forecast", zorder=5)
    ax2.axvline(x=selected_year, linestyle="--", alpha=0.5)

ax2.set_xlabel("Year")
ax2.set_ylabel("Number of Deaths")
ax2.set_title("Under-5 Deaths with Forecast")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)  # Show combined historical + forecast chart
