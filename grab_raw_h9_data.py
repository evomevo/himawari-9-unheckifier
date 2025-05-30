import os, boto3, botocore, bz2
from datetime import datetime, timedelta, timezone

def auto_download_latest_segments(band='B13', res='R20', flv='FLDK', hours_back=24):
    """
    Searches back up to `hours_back` hours (in 10-minute steps) for the
    most recent Himawari-9 .DAT.bz2 segments, downloads and decompresses them.
    """
    bucket = 'noaa-himawari9'
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    now = datetime.now(timezone.utc)
    # Round down to nearest 10 minutes
    base_time = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 10)
    max_attempts = hours_back * 6  # 6 ten-minute intervals per hour

    for i in range(max_attempts):
        ts = base_time - timedelta(minutes=10 * i)
        yyyy, mm, dd = ts.strftime('%Y'), ts.strftime('%m'), ts.strftime('%d')
        hhmm = ts.strftime('%H%M')
        prefix = f'AHI-L1b-FLDK/{yyyy}/{mm}/{dd}/{hhmm}/'

        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in resp:
                continue
        except botocore.exceptions.ClientError:
            continue

        # Filter matching segments
        seg_keys = [
            obj['Key'] for obj in resp['Contents']
            if obj['Key'].endswith('.DAT.bz2')
            and f'_{band}_' in obj['Key']
            and f'_{res}_' in obj['Key']
            and f'_{flv}_' in obj['Key']
        ]

        if seg_keys:
            # Found the latest time with segments
            local_dir = os.path.join('himawari9_raw', f'{yyyy}{mm}{dd}_{hhmm}')
            os.makedirs(local_dir, exist_ok=True)
            print(f"Found segments for {yyyy}-{mm}-{dd} {hhmm} UTC, downloading...")

            for key in sorted(seg_keys):
                fname = os.path.basename(key)
                local_bz2 = os.path.join(local_dir, fname)
                s3.download_file(bucket, key, local_bz2)
                # decompress
                local_dat = local_bz2[:-4]
                with bz2.open(local_bz2, 'rb') as f_in, open(local_dat, 'wb') as f_out:
                    f_out.write(f_in.read())
                print(f"  → {fname} downloaded and decompressed.")

            print(f"All segments saved to: {local_dir}")
            return local_dir

    print(f"No segments found in the last {hours_back} hours.")
    return None

if __name__ == '__main__':
    auto_download_latest_segments(hours_back=24)