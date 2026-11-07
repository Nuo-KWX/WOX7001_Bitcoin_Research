
import requests
import zipfile
import io
import os
import pandas as pd

# 1. Set project folder
DATA_DIR = "data/raw/binance"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Set Bitcoin data period
months = pd.period_range(
    start="2021-01",
    end="2025-12",
    freq="M"
)

all_data = []

# 3. Download Binance daily Bitcoin data
for month in months:

    month_str = str(month)

    url = (
        "https://data.binance.vision/data/spot/monthly/klines/"
        f"BTCUSDT/1d/BTCUSDT-1d-{month_str}.zip"
    )

    print(f"Downloading {month_str}...")

    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            print(f"File unavailable: {month_str}")
            continue

        # Open downloaded ZIP file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:

            csv_file = z.namelist()[0]

            with z.open(csv_file) as f:
                df = pd.read_csv(f, header=None)

        all_data.append(df)

    except Exception as e:
        print(f"Error downloading {month_str}: {e}")

# 4. Combine monthly data
if all_data:

    bitcoin = pd.concat(all_data, ignore_index=True)

    # 5. Assign column names
    bitcoin.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    # 6. Save combined CSV
    output_path = os.path.join(
        DATA_DIR,
        "BTCUSDT_Daily_2021_2025.csv"
    )

    bitcoin.to_csv(output_path, index=False)

    print("\nDownload completed!")
    print("Total rows:", len(bitcoin))
    print("Saved to:", output_path)

else:
    print("No data downloaded. Please check your internet connection.")
