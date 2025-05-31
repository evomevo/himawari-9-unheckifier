import os, boto3, botocore
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

band = '13'
hours_back = 24

# ─────────────────────────────────────────────────────────────────────────────
# 1) AUTO-DOWNLOAD LATEST SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────

def auto_download_latest_segments(band=band, hours_back=hours_back):
    product = 'ABI-L1b-RadF' # Full disk radiance
    bucket = 'noaa-goes19'
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    max_attempts = hours_back # 1-hour intervals

    for i in range(max_attempts):
        ts = now - timedelta(hours=i)
        year = ts.strftime('%Y')
        doy = ts.strftime('%j') # Day of year
        hour = ts.strftime('%H')

        prefix = f'{product}/{year}/{doy}/{hour}/'

        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in resp:
                continue
        except botocore.exceptions.ClientError:
            continue

        files = [
            obj['Key'] for obj in resp['Contents']
            if f'-M6C{band:0>2}' in obj['Key'] and obj['Key'].endswith('.nc')
        ]

        latest = sorted(files)[-1]


        if latest:
            local_dir = os.path.join('goes19_raw', f'{year}{doy}_{hour}00')
            os.makedirs(local_dir, exist_ok=True)
            print(f"Found GOES-19 Full Disk file for {year}-{doy} {hour} UTC")

            local_file = os.path.join(local_dir, os.path.basename(latest))
            s3.download_file(bucket, latest, local_file)
            print(f"   Downloaded {os.path.basename(latest)}")

            return local_dir

    print(f"No segments found in the last {hours_back} hours.")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 2) RUN THE AUTO-DOWNLOAD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    auto_download_latest_segments()