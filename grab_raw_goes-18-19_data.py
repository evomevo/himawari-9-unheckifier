import os, boto3, botocore
from datetime import datetime, timedelta, timezone

# -----------------------------------------------------------------------------
# 0) CONFIGURATION
# -----------------------------------------------------------------------------

hours_back = 6  # How many hours back to search for segments
product = 'ABI-L1b-RadF'  # GOES ABI Level 1b Radiance Full Disk
band = 13

# Configuration for both satellites
satellites = [
    {
        "name": "goes-19",
        "bucket": "noaa-goes19",
        "product": product,
        "band": band
    },
    {
        "name": "goes-18",
        "bucket": "noaa-goes18",
        "product": product,
        "band": band
    }
]

# -----------------------------------------------------------------------------
# 1) AUTO-DOWNLOAD LATEST SEGMENTS FUNCTION
# -----------------------------------------------------------------------------

def auto_download_latest_goes(sat_dict):
    """
    Searches back up to `hours_back` hours (in 1-hour steps) for the most recent
    full-globe .nc composite of a given GOES satellite, downloads it, and returns the local directory.
    """

    bucket = sat_dict["bucket"]
    product = sat_dict["product"]
    band = sat_dict["band"]
    name = sat_dict["name"]

    # Use unsigned S3 client for public buckets.
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    max_attempts = hours_back # 1-hour intervals

    for i in range(max_attempts):
        ts = now - timedelta(hours=i)
        year = ts.strftime('%Y')
        doy = ts.strftime('%j') # Day of year
        hour = ts.strftime('%H')

        # S3 prefix: {product}/{year}/{day-of-year}/{hour}/
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

        if not files:
            continue

        latest = sorted(files)[-1]

        local_dir = os.path.join(f"{name}_raw", f'{year}{doy}_{hour}00')
        os.makedirs(local_dir, exist_ok=True)
        print(f"Found {name.upper()} Full Disk file for {year}-{doy} {hour} UTC")

        local_file = os.path.join(local_dir, os.path.basename(latest))
        s3.download_file(bucket, latest, local_file)
        print(f"   Downloaded {os.path.basename(latest)} to {local_dir}")

        return local_dir

    print(f"No files found in the last {hours_back} hours in {bucket}/{product}.")
    return None

# -----------------------------------------------------------------------------
# 2) RUN THE AUTO-DOWNLOAD FUNCTION FOR BOTH SATELLITES
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    for sat in satellites:
        auto_download_latest_goes(sat)