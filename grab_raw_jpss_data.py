import os, boto3, botocore
from datetime import datetime, timedelta, timezone

# -----------------------------------------------------------------------------
# 0) CONFIGURATION
# -----------------------------------------------------------------------------

bucket = 'noaa-nesdis-n20-pds'
band = 'M15'
product = f'VIIRS-{band}-SDR'
days_back = 2

def auto_download_latest_viirs():
    """
    Searches up to `days_back` days for the most recent
    NOAA-20 VIIRS SDR .h5 files, downloads them, and returns the local directory.
    """
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
    now = datetime.now(timezone.utc)

    for i in range(days_back):
        dt = now - timedelta(days=i)
        year = dt.strftime("%Y")
        month = dt.strftime("%m")
        day = dt.strftime("%d")
        prefix = f"{product}/{year}/{month}/{day}/"

        print(f"Checking {prefix}")
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        except botocore.exceptions.ClientError as e:
            print(f"  Error accessing {prefix}: {e}")
            continue

        if 'Contents' in response:
            local_dir = os.path.join('noaa-20_viirs_raw', f'{year}{month}{day}')
            os.makedirs(local_dir, exist_ok=True)
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                if not filename:
                    continue
                local_file = os.path.join(local_dir, filename)
                if not os.path.exists(local_file):
                    s3.download_file(bucket, key, local_file)
                    print(f"   Downloaded {filename} to {local_dir}")
        else:
            print(f"   No files found for the last {days_back} days in {prefix}.")

# -----------------------------------------------------------------------------
# 2) RUN THE AUTO-DOWNLOAD FUNCTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    auto_download_latest_viirs()