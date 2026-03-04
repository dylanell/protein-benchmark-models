"""S3-compatible I/O utilities."""

import os


def get_storage_options(path: str = "") -> dict:
    """Return S3 storage options for pandas.

    Returns a dict suitable for passing to pandas read_csv/to_csv as
    storage_options. Only returns credentials when path is an S3 path,
    so local paths work transparently even when S3 env vars are set.
    """
    if not path.startswith("s3://"):
        return {}
    return {
        "endpoint_url": os.environ["S3_ENDPOINT_URL"],
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    }


def get_s3_filesystem():
    """Return a configured s3fs.S3FileSystem instance."""
    import s3fs
    return s3fs.S3FileSystem(
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
