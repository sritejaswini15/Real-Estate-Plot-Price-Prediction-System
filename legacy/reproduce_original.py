"""Re-run the original project's pipeline to document what it actually did.

    python legacy/reproduce_original.py

1. Re-applies the original EDA.py cleaning to the raw export and checks that it
   reproduces legacy/cleaned_data.csv exactly.
2. Re-runs Model_Evaluation.py (LinearRegression on get_dummies of every
   column, including Title and Location, with one random 80/20 split).
3. Calls Price_Prediction.py's predict_price() with different inputs. It passes
   land_area / no_of_open_gates / road_width, but the model was trained on
   Plot_Area / open_sides / Width_road. reindex() therefore fills the real
   features with 0: plot area, open sides and road width are ignored, and the
   output is the same (negative) number for every plot of a given facing.
   The original hid the sign with abs().

The code below copies the original logic. Only file paths changed: the
originals read from Google Sheets / Google Drive in Colab.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent


def clean_price(value):  # copied from EDA.py
    if isinstance(value, str):
        value = value.strip().lower()
        if "crore" in value:
            return float(value.replace("crore", "").replace("c", "").strip()) * 1e7
        elif "lac" in value:
            return float(value.replace("lac", "").replace("l", "").strip()) * 1e5
        elif "ac" in value:
            return float(value.replace("ac", "").replace("a", "").strip()) * 1e4
        else:
            return pd.to_numeric(value, errors="coerce")
    return value


def main() -> None:
    raw = pd.read_excel(ROOT / "data" / "raw" / "99acres_patancheru_properties.xlsx")
    df = raw.dropna().drop_duplicates(keep="first")
    df["Price"] = df["Price"].apply(clean_price)
    df["open_sides"] = pd.to_numeric(df["open_sides"], errors="coerce")
    df["Width_road"] = pd.to_numeric(df["Width_road"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    df = df.reset_index(drop=True)
    df["Price"] = df["Price"].round().astype("int64")
    df["Width_road"] = df["Width_road"].astype("int64")
    original = pd.read_csv(Path(__file__).with_name("cleaned_data.csv"))
    same = (df.values.astype(str) == original.values.astype(str)).all()
    print(f"1. Original cleaning keeps {len(df)} of {len(raw)} rows; identical to cleaned_data.csv: {same}")

    data = original.dropna()
    X = pd.get_dummies(data.drop(columns=["Price"]), drop_first=True)
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    print(f"2. Original evaluation: {X.shape[1]} features for {len(X_train)} training rows; "
          f"R² {r2_score(y_test, pred):.4f}, MAE ₹{mean_absolute_error(y_test, pred):,.0f}, MAPE {mape:.2f}% "
          f"on {len(y_test)} test rows")

    def predict_price(area_sq_yards, facing_input, open_sides, road_width):  # copied from Price_Prediction.py
        land_area = area_sq_yards * 9
        facing_columns = ["facing_North", "facing_East", "facing_South", "facing_West", "facing_North-East",
                          "facing_North-West", "facing_South-East", "facing_South-West"]
        facing_vector = {col: 0 for col in facing_columns}
        if f"facing_{facing_input}" in facing_vector:
            facing_vector[f"facing_{facing_input}"] = 1
        input_df = pd.DataFrame([{"land_area": land_area, "no_of_open_gates": open_sides,
                                  "road_width": road_width, **facing_vector}])
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        return model.predict(input_df)[0]

    used = [c for c in ("land_area", "no_of_open_gates", "road_width") if c in model.feature_names_in_]
    print(f"3. Inputs the trained model actually receives from predict_price: {used or 'none'}")
    for args in [(150, "East", 2, 30), (8000, "East", 2, 30), (150, "East", 1, 100), (150, "West", 2, 30)]:
        value = predict_price(*args)
        print(f"   predict_price{args} = ₹{value:,.0f}  -> shown to the user as ₹{abs(value):,.2f}")


if __name__ == "__main__":
    main()
