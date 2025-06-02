import os, boto3, botocore
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

bucket = 'noaa-gmgsi-pds'
hours_back = 6  # How many hours back to search (in whole-hour steps)

# ─────────────────────────────────────────────────────────────────────────────
# 1) AUTO-DOWNLOAD LATEST GMGSI COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────

def auto_download_latest_gmgsi():
    """
    Searches back up to `hours_back` hours for the most recent
    GMGSI full-globe .nc composite, downloads it, and returns the local directory.
    """
    
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    for i in range(hours_back):
        ts = now - timedelta(hours=i)
        year = ts.strftime('%Y')
        month = ts.strftime('%m')
        day = ts.strftime('%d')
        hour = ts.strftime('%H')

        # GMGSI_LW/YYYY/MM/DD/HH/
        prefix = f'GMGSI_LW/{year}/{month}/{day}/{hour}/'

        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in resp:
                continue
        except botocore.exceptions.ClientError:
            continue

        # Collect all .nc keys under that prefix
        nc_keys = [
            obj['Key'] for obj in resp['Contents']
            if obj['Key'].endswith('.nc')
        ]

        if nc_keys:
            # Pick the lexicographically latest filename (usually corresponds to the latest timestamp)
            latest_key = sorted(nc_keys)[-1]

            local_dir = os.path.join('gmgsi_raw', f'{year}{month}{day}_{hour}00')
            os.makedirs(local_dir, exist_ok=True)
            print(f"Found GMGSI composite for {year}-{month}-{day} {hour} UTC -> {os.path.basename(latest_key)}")

            local_file = os.path.join(local_dir, os.path.basename(latest_key))
            s3.download_file(bucket, latest_key, local_file)
            print(f"   Downloaded {os.path.basename(latest_key)} into {local_dir}")

            return local_dir

    print(f"No GMGSI composites found in the last {hours_back} hours.")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 2) RUN THE AUTO-DOWNLOAD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    auto_download_latest_gmgsi()