import tempfile
from pathlib import Path
from typing import Any, Optional

import molflux.modelzoo as mz


def load_from_dvc(repo_url: str, rev: str, model_path_in_repo: str) -> Any:
    try:
        from dvc.api import DVCFileSystem  # pyright: ignore
    except ImportError as err:
        from molflux.modelzoo.errors import OptionalDependencyImportError

        raise OptionalDependencyImportError("DVC", "dvc[s3]") from err
    else:
        fs = DVCFileSystem(repo_url, rev=rev, subrepos=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            fs.download(
                model_path_in_repo,
                str(Path(tmpdir) / "model"),
                recursive=True,
            )
            model = mz.load_from_store(Path(tmpdir) / "model")

        return model


def load_from_dvc_or_disk(
    path: Optional[str] = None,
    repo_url: Optional[str] = None,
    rev: Optional[str] = None,
    model_path_in_repo: Optional[str] = None,
) -> Any:
    if path is not None:
        model = mz.load_from_store(path)
    elif (
        (repo_url is not None)
        and (rev is not None)
        and (model_path_in_repo is not None)
    ):
        model = load_from_dvc(
            repo_url=repo_url,
            rev=rev,
            model_path_in_repo=model_path_in_repo,
        )
    else:
        raise ValueError(
            "Must specify either 'path' or all of 'repo_url', 'rev', 'model_path_in_repo' (but not both).",
        )

    return model
