"""Optional file transport only; uses the notebook/instance IAM role."""

import zipfile
from pathlib import Path
from urllib.parse import urlparse


def s3_parts(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("Expected s3://bucket/key")
    return parsed.netloc, parsed.path.lstrip("/")


def download(uri, destination):
    import boto3
    bucket, key = s3_parts(uri)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination)+".part")
    boto3.client("s3").download_file(bucket, key, str(temporary))
    temporary.replace(destination)


def upload(path, uri):
    import boto3
    bucket, key = s3_parts(uri)
    boto3.client("s3").upload_file(str(path), bucket, key)


def extract_resource(archive, destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if i.filename.startswith("student_resource/") and not i.filename.endswith(".DS_Store")]
        for member in members:
            target = (root/member.filename).resolve()
            if not target.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP member")
        z.extractall(root, members)
    return root/"student_resource"/"dataset"
