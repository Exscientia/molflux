import functools
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import botocore.session
import pyarrow as pa
from boto3.session import Session
from botocore.credentials import RefreshableCredentials
from dateutil.tz import tzlocal
from fsspec.callbacks import _DEFAULT_CALLBACK
from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow.fs import AwsStandardS3RetryStrategy, S3RetryStrategy

logger = logging.getLogger(__name__)


class S3FileSystem(ArrowFSWrapper):
    """A fsspec compliant S3 FileSystem interface, backed by pyarrow.

    References:
        https://arrow.apache.org/docs/python/filesystems.html#using-arrow-filesystems-with-fsspec
        https://arrow.apache.org/docs/python/generated/pyarrow.fs.S3FileSystem.html
    """

    root_marker = ""
    protocol = "s3"

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        scheme: Optional[str] = None,
        endpoint_override: Optional[str] = None,
        background_writes: bool = True,
        default_metadata: Optional[Union[Dict, pa.KeyValueMetadata]] = None,
        role_arn: Optional[str] = None,
        session_name: Optional[str] = None,
        external_id: Optional[str] = None,
        load_frequency: int = 900,
        proxy_options: Optional[Union[str, Dict]] = None,
        allow_bucket_creation: bool = False,
        allow_bucket_deletion: bool = False,
        botocore_session: Optional[botocore.session.Session] = None,
        boto3_session: Optional[Session] = None,
        retry_strategy: Optional[S3RetryStrategy] = None,
        session_ttl: int = 60 * 60,
        refresh_in: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if retry_strategy is None:
            retry_strategy = AwsStandardS3RetryStrategy(max_attempts=3)

        self._refresh_in = refresh_in

        # setup boto3 session for properly fetching credentials from AWS
        self._session = get_refreshable_boto3_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            profile_name=session_name,
            botocore_session=botocore_session,
            boto3_session=boto3_session,
            session_ttl=session_ttl,
        )

        # pyarrow.fs.S3FileSystem initialiser with only aws credentials as free parameters
        self._pyarrow_fs_S3FileSystem = functools.partial(
            pa.fs.S3FileSystem,
            anonymous=False,
            role_arn=role_arn,
            session_name=session_name,
            external_id=external_id,
            load_frequency=load_frequency,
            region=self._session.region_name,
            scheme=scheme,
            endpoint_override=endpoint_override,
            background_writes=background_writes,
            default_metadata=default_metadata,
            proxy_options=proxy_options,
            allow_bucket_creation=allow_bucket_creation,
            allow_bucket_deletion=allow_bucket_deletion,
            retry_strategy=retry_strategy,
        )

        self._fs = self._get_refreshed_filesystem()
        super().__init__(fs=self._fs, **kwargs)

    def _get_refreshed_filesystem(self) -> pa.fs.S3FileSystem:
        """Returns a filesystem with refreshed AWS credentials."""
        logger.debug("Refreshing AWS credentials...")
        credentials = self._session.get_credentials().get_frozen_credentials()
        return self._pyarrow_fs_S3FileSystem(
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            session_token=credentials.token,
        )

    @property
    def refresh_needed(self) -> bool:
        """ "Checks if a credentials refresh is needed.

        By default, credentials will be refreshed if they are going to expire
        within the next 15 minutes.

        References:
            https://github.com/boto/botocore/blob/develop/botocore/credentials.py#L438
        """
        botocore_session = self._session._session
        refreshable_credentials = botocore_session._credentials
        return refreshable_credentials.refresh_needed(refresh_in=self._refresh_in)  # type: ignore[no-any-return]

    @property
    def fs(self) -> pa.fs.S3FileSystem:
        """Returns a filesystem with refreshed credentials (if needed).

        This means that every filesystem operation will check if credentials
        are fresh before actuating - as internally self.fs is used to perform
        such operations.
        """
        if self.refresh_needed:
            self._fs = self._get_refreshed_filesystem()
        return self._fs

    @fs.setter
    def fs(self, value: pa.fs.S3FileSystem) -> None:
        self._fs = value

    def get_file(
        self,
        rpath: Any,
        lpath: Any,
        callback: Any = _DEFAULT_CALLBACK,
        outfile: Any = None,
        **kwargs: Any,
    ) -> None:
        """Copy single remote file to local.

        We re-implement this method here to fix a bug when interacting with
        the datasets library and their use of tqdm. The code below is vendored
        from the original base class implementation, with minimal changes.

        References:
            https://github.com/fsspec/filesystem_spec/blob/45a6aec7da1407243f9767c6ab0cff40efee72eb/fsspec/spec.py#L868
            https://arrow.apache.org/docs/python/generated/pyarrow.NativeFile.html#pyarrow.NativeFile.size
        """
        from fsspec.implementations.local import LocalFileSystem
        from fsspec.utils import isfilelike

        if isfilelike(lpath):
            outfile = lpath
        elif self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return None

        LocalFileSystem(auto_mkdir=True).makedirs(self._parent(lpath), exist_ok=True)

        with self.open(rpath, "rb", **kwargs) as f1:
            if outfile is None:
                outfile = open(lpath, "wb")

            try:
                # for pyarrow files "size" is a callable, not a property
                # so replaced getattr(f1, "size", None) with the following
                size = f1.size() if hasattr(f1, "size") else None
                callback.set_size(size=size)
                data = True
                while data:
                    data = f1.read(self.blocksize)
                    segment_len = outfile.write(data)
                    if segment_len is None:
                        segment_len = len(data)  # type: ignore[arg-type]
                    callback.relative_update(segment_len)
            finally:
                if not isfilelike(lpath):
                    outfile.close()


def get_refreshable_boto3_session(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
    botocore_session: Optional[botocore.session.Session] = None,
    boto3_session: Optional[Session] = None,
    session_ttl: int = 60 * 60,
) -> Session:
    """Returns a boto3 session with automatically refreshing credentials.

    Examples:
        >>> session = get_refreshable_boto3_session()
        >>> client = session.client("s3") # we can cache this client object without worrying about expiring credentials
        >>> credentials = session.get_credentials().get_frozen_credentials()  # or we can use it to generate new credentials

    Args:
        session_ttl: The number of seconds to request a session for. Defaults to
            1 hour as that is the maximum session duration allowed by IAM
            profiles in most cases.

    Returns:
        An automatically refreshing boto3 session.

    References:
        https://dev.to/li_chastina/auto-refresh-aws-tokens-using-iam-role-and-boto3-2cjf
        https://stackoverflow.com/a/69226170
    """

    def _get_session_credentials() -> Dict[str, Any]:
        """The refresh callable."""
        if boto3_session is not None:
            session = boto3_session
        else:
            session = Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=aws_region_name,
                profile_name=profile_name,
                botocore_session=botocore_session,
            )

        credentials = {
            **session.get_credentials().get_frozen_credentials()._asdict(),
            "expiry_time": (
                datetime.now(tzlocal()) + timedelta(seconds=session_ttl)
            ).isoformat(),
        }

        logger.debug("AWS credentials expiry time: %s", credentials["expiry_time"])
        return credentials

    # create session with refreshable credentials
    refreshable_credentials = RefreshableCredentials.create_from_metadata(
        metadata=_get_session_credentials(),
        refresh_using=_get_session_credentials,
        method="sts-assume-role",
    )
    refreshable_botocore_session = botocore.session.get_session()
    refreshable_botocore_session._credentials = refreshable_credentials
    refreshable_botocore_session.set_config_variable(
        "region",
        aws_region_name or os.environ.get("AWS_REGION"),
    )
    refreshable_boto3_session = Session(botocore_session=refreshable_botocore_session)
    return refreshable_boto3_session
