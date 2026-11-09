import pandas as pd
from mcp.server.fastmcp import FastMCP
from xgboost import XGBRegressor

H = 28
df = pd.read_csv("train.csv", parse_dates=["date"])
mcp = FastMCP("sales-forecast")

def build_features(s):
    d = s.to_frame("sales")
    for lag in (28, 35, 42, 364):
        d[f"lag_{lag}"] = s.shift(lag)
    d["roll28"] = s.shift(H).rolling(28).mean()
    d["weekday"] = d.index.day_of_week
    d["month"] = d.index.month
    d["doy"] = d.index.day_of_year
    return d

@mcp.tool()
def forecast_sales(store: int, item: int, weeks: int = 4) -> dict:
    """Forecast daily sales for one store (1-10) and one item (1-50) for the next 1 to 4 weeks."""
    if not 1 <= weeks <= 4:
        return {"error": "weeks must be between 1 and 4"}
    s = df[(df["store"] == store) & (df["item"] == item)].set_index("date")["sales"]
    if s.empty:
        return {"error": "unknown store or item (store 1-10, item 1-50)"}

    future = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=weeks * 7)
    d = build_features(s.reindex(s.index.append(future)))     # history plus empty future days
    feats = [c for c in d.columns if c != "sales"]

    train = d.loc[s.index].dropna()
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(train[feats], train["sales"])
    pred = model.predict(d.loc[future, feats])

    return {"store": store, "item": item,
            "daily_forecast": {str(day.date()): round(float(p), 1) for day, p in zip(future, pred)},
            "weekly_totals": [round(float(pred[i:i + 7].sum())) for i in range(0, len(pred), 7)]}

if __name__ == "__main__":
    mcp.run()
