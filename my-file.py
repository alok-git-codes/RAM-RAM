!pip install yfinance pandas numpy --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# TARGET PATTERN
# ==================================================

TARGET_SYMBOL = "ANGELONE.NS"

print("📡 Angel One Pattern Loading...")

target_df = yf.download(
    TARGET_SYMBOL,
    start="2025-01-07",
    end="2025-03-20",
    auto_adjust=True,
    progress=False
)

if target_df.empty:
    raise Exception("Angel One data not found!")

if isinstance(target_df.columns, pd.MultiIndex):
    target_df.columns = target_df.columns.get_level_values(0)

target_close = target_df["Close"].values

target_norm = (
    target_close - target_close.min()
) / (
    target_close.max() - target_close.min()
)

pattern_length = len(target_norm)

print(f"✅ Pattern Length = {pattern_length} candles")


# ==================================================
# NIFTY 500 LIST
# ==================================================

url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

nifty500 = pd.read_csv(url)

tickers = [
    s + ".NS"
    for s in nifty500["Symbol"]
]

print(f"🔎 Total Stocks: {len(tickers)}")


# ==================================================
# MATCH FUNCTION
# ==================================================

def get_match_score(base, sample):

    corr = np.corrcoef(
        base,
        sample
    )[0, 1]

    mae = np.mean(
        np.abs(base - sample)
    )

    return corr, mae


# ==================================================
# SCAN
# ==================================================

results = []

for symbol in tickers:

    try:

        df = yf.download(
            symbol,
            period="250d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].values

        if len(close) < pattern_length:
            continue

        best_corr = -999
        best_mae = 999
        best_date = None

        # Sliding Window Search

        for i in range(
            len(close) - pattern_length + 1
        ):

            window = close[
                i:i + pattern_length
            ]

            if np.max(window) == np.min(window):
                continue

            window_norm = (
                window - np.min(window)
            ) / (
                np.max(window) - np.min(window)
            )

            corr, mae = get_match_score(
                target_norm,
                window_norm
            )

            if corr > best_corr:

                best_corr = corr
                best_mae = mae

                best_date = df.index[
                    i + pattern_length - 1
                ]

        if (
            best_corr >= 0.85 and
            best_mae <= 0.15
        ):

            results.append({

                "Symbol": symbol,

                "Corr": round(
                    best_corr,
                    3
                ),

                "MAE": round(
                    best_mae,
                    4
                ),

                "Date": best_date

            })

            print(
                f"✅ {symbol} | "
                f"Corr={best_corr:.3f} | "
                f"MAE={best_mae:.4f} | "
                f"Date={best_date.date()}"
            )

    except Exception:
        continue


# ==================================================
# FINAL RESULT
# ==================================================

print("\n")
print("=" * 70)
print("🎯 RECENT MATCHES")
print("=" * 70)

if len(results):

    result_df = pd.DataFrame(results)

    result_df["Date"] = pd.to_datetime(
        result_df["Date"]
    )

    # सबसे नई तारीख ऊपर
    result_df = result_df.sort_values(
        by="Date",
        ascending=False
    )

    print("\n")

    for _, row in result_df.iterrows():

        print(
            f"{row['Symbol']:15} "
            f"Corr={row['Corr']:.3f}   "
            f"MAE={row['MAE']:.4f}   "
            f"Date={row['Date'].date()}"
        )

else:

    print("❌ No Match Found")
