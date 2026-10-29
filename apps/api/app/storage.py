"""Presigned URLs for object storage. The API never moves audio bytes itself (PLAN.md §5)."""
from functools import lru_cache

import botocore.session
from botocore.config import Config

from app.settings import get_settings


class Storage:
    def __init__(self):
        s = get_settings()
        self.bucket = s.s3_bucket
        self._s3 = botocore.session.get_session().create_client(
            "s3", region_name=s.s3_region, endpoint_url=s.s3_endpoint or None,
            aws_access_key_id=s.s3_access_key or None, aws_secret_access_key=s.s3_secret_key or None,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))

    def presign_put(self, key: str, content_type: str, expires: int) -> str:
        return self._s3.generate_presigned_url(
            "put_object", Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires)

    def presign_get(self, key: str, expires: int) -> str:
        return self._s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires)


@lru_cache
def get_storage() -> Storage:
    """FastAPI dependency (tests override it with a fake)."""
    return Storage()
