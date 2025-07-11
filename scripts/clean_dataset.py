import pandas as pd, pathlib, argparse

def main(src, dest):
    # 1 – load CSV, treat Timestamp as datetime
    df = pd.read_csv(src, parse_dates=['Timestamp'])

    # 2 – standardise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 3 – optional rename for convenience
    df = df.rename(columns={
        'logistics_delay_reason': 'delay_reason',
        'logistics_delay':        'delay_flag',
        'shipment_status':        'shipment_status'
    })

    # 4 – derive delivered flag
    df['is_delivered'] = df['shipment_status'].str.lower().eq('delivered')

    # 5 – save to Parquet
    pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src',  default='data/raw/smart_logistics_dataset.csv')
    p.add_argument('--dest', default='data/processed/shipments.parquet')
    main(**vars(p.parse_args()))
