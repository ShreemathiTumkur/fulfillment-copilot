import pandas as pd, pathlib, argparse

def main(src, dest):
    df = pd.read_csv(src, parse_dates=['Ship_Date', 'Delivery_Date'])
    df = df.dropna(subset=['Shipment_ID', 'Ship_Date', 'Delivery_Date'])
    df['otif'] = (df['Status'].str.lower() == 'delivered') & \
                 ((df['Delivery_Date'] - df['Ship_Date']).dt.days <= 0)
    df.columns = [c.lower().strip() for c in df.columns]
    pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--src',  default='data/raw/smart_logistics_dataset.csv')
    p.add_argument('--dest', default='data/processed/shipments.parquet')
    main(**vars(p.parse_args()))

