import os, boto3, botocore, bz2

# -----------------------------------------------------------------------------
# Himawari-8 has been discontinued.
# Grab the data you want while it's still available.
# -----------------------------------------------------------------------------
# 0) CONFIGURATION
# -----------------------------------------------------------------------------

bucket = 'noaa-himawari8'
product = 'AHI-L1b-FLDK'
band = 'B13' # Up to 16 bands available for Himawari-8/9: B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B13, B14, B15, B16
res = 'R20'
flv = 'FLDK'
# As Himawari-8 has been discontinued, you have to specify a fixed date and time up until December 13, 2022, at 04:50 UTC.
year = '2022'
month = '12'
day = '13'
hhmm = '0450'

# -----------------------------------------------------------------------------
# 1) AUTO-DOWNLOAD LATEST SEGMENTS
# -----------------------------------------------------------------------------

def auto_download_latest_himawari():
    """
    Searches back up to `hours_back` hours (in 10-minute steps) for the most recent
    Himawari-8 .DAT.bz2 segments, downloads, decompresses them, and returns the local directory.
    """
    
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))


    prefix = f'{product}/{year}/{month}/{day}/{hhmm}/'
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

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
        local_dir = os.path.join('himawari-8_raw', f'{year}{month}{day}_{hhmm}')
        os.makedirs(local_dir, exist_ok=True)
        print(f"Found segments for {year}-{month}-{day} {hhmm} UTC, downloading...")
        
        for key in sorted(seg_keys):
            fname = os.path.basename(key)
            local_bz2 = os.path.join(local_dir, fname)
            s3.download_file(bucket, key, local_bz2)
            # Decompress
            local_dat = local_bz2[:-4]
            with bz2.open(local_bz2, 'rb') as f_in, open(local_dat, 'wb') as f_out:
                f_out.write(f_in.read())
            # Delete the .bz2 compressed file to save space. Remove this line if you want to keep it.
            os.remove(local_bz2)
            print(f"   Downloaded {fname} and decompressed into {local_dir}.")

        print(f"All segments saved to: {local_dir}")
        return local_dir

    print(f"No segments found in {bucket}/{product}.")
    return None

# -----------------------------------------------------------------------------
# 2) RUN THE AUTO-DOWNLOAD FUNCTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    auto_download_latest_himawari()