import pathlib
import tempfile

import nox

SUPPORTED_PYTHON_VERSIONS = ["3.10", "3.11"]

nox.options.envdir = ".cache/nox"
nox.options.reuse_existing_virtualenvs = False
nox.options.default_venv_backend = "uv"
nox.options.sessions = [
    "formatting_check-3.11",
    "typing_check-3.11",
    "tests_run_latest-3.11",
]

SOURCE_FILES = ("src/", "tests/", "noxfile.py")

# Artefacts folders
PINNED_VERSIONS = "pinned-versions"
COVERAGE_DIR = ".coverage"

TEST_REPORTS_DIR = "test-reports"

# all external utility tools used by our nox sessions
BUILD_TOOLS = ["build"]
COVERAGE_TOOLS = [
    "coverage[toml]",
    "coverage-badge",
    "setuptools",  # setuptools required with uv as coverage-badge imports pkg_resources
]
RUFF_VERSION = "ruff~=0.4"

SUBMODULE_EXTRAS: dict[str, dict[str, list | None]] = {
    "core": {
        "doctests_target_dir_core": [
            # "src/molflux/core"
        ],
        "tests_target_dir_core": ["tests/core/core"],
        "doctests_target_dir_extras": None,
        "tests_target_dir_extras": None,
        "extras": [None],
    },
    "datasets": {
        "doctests_target_dir_core": [
            # "src/molflux/datasets"
        ],
        "tests_target_dir_core": ["tests/datasets/core", "tests/datasets/plugins/core"],
        "doctests_target_dir_extras": [
            # "src/molflux/datasets/builders"
        ],
        "tests_target_dir_extras": ["tests/datasets/plugins"],
        "extras": [None, "openeye", "rdkit", "tdc"],
    },
    "features": {
        "doctests_target_dir_core": [
            # "src/molflux/features",
            # "src/molflux/features/representations/core",
        ],
        "tests_target_dir_core": ["tests/features/core", "tests/features/plugins/core"],
        "doctests_target_dir_extras": [
            # "src/molflux/features/representations"
        ],
        "tests_target_dir_extras": ["tests/features/plugins"],
        "extras": [None, "openeye", "rdkit"],
    },
    "splits": {
        "doctests_target_dir_core": [
            # "src/molflux/splits",
            # "src/molflux/splits/strategies/core",
        ],
        "tests_target_dir_core": ["tests/splits/core", "tests/splits/plugins/core"],
        "doctests_target_dir_extras": [
            # "src/molflux/splits/strategies"
        ],
        "tests_target_dir_extras": ["tests/splits/plugins"],
        "extras": [None, "openeye", "rdkit"],
    },
    "modelzoo": {
        "doctests_target_dir_core": [
            # "src/molflux/modelzoo",
            # "src/molflux/modelzoo/models/core",
        ],
        "tests_target_dir_core": ["tests/modelzoo/core", "tests/modelzoo/plugins/core"],
        "doctests_target_dir_extras": [
            # "src/molflux/modelzoo/models"
        ],
        "tests_target_dir_extras": ["tests/modelzoo/plugins"],
        "extras": [
            None,
            "catboost",
            "ensemble",
            "lightning",
            "mapie",
            "pyod",
            "pystan",
            "sklearn",
            "xgboost",
        ],
    },
    "metrics": {
        "doctests_target_dir_core": [
            # "src/molflux/metrics"
        ],
        "tests_target_dir_core": ["tests/metrics/core", "tests/metrics/plugins/core"],
        "doctests_target_dir_extras": None,
        "tests_target_dir_extras": None,
        "extras": [None],
    },
}

EXTRAS: list[str | None] = []
for v in SUBMODULE_EXTRAS.values():
    EXTRAS += v["extras"]  # type: ignore

EXTRAS = list(set(EXTRAS))

# skip openeye in github actions (cannot pip install)
EXTRAS.remove("openeye")

INVERTED_SUBMODULE_EXTRAS = {}
for extra in EXTRAS:
    doctests_target_dir_core = []
    tests_target_dir_core = []
    doctests_target_dir_extras = []
    tests_target_dir_extras = []
    for v in SUBMODULE_EXTRAS.values():
        if extra in v["extras"]:  # type: ignore
            if v["doctests_target_dir_core"]:
                doctests_target_dir_core += v["doctests_target_dir_core"]
            if v["tests_target_dir_core"]:
                tests_target_dir_core += v["tests_target_dir_core"]
            if v["doctests_target_dir_extras"]:
                doctests_target_dir_extras += v["doctests_target_dir_extras"]
            if v["tests_target_dir_extras"]:
                tests_target_dir_extras += v["tests_target_dir_extras"]

    INVERTED_SUBMODULE_EXTRAS[extra] = {
        "doctests_target_dir_core": doctests_target_dir_core,
        "tests_target_dir_core": tests_target_dir_core,
        "doctests_target_dir_extras": doctests_target_dir_extras,
        "tests_target_dir_extras": tests_target_dir_extras,
    }


def resolve_lockfile_path(
    python_version: str,
    extra: str | None = None,
    rootdir: str = PINNED_VERSIONS,
) -> pathlib.Path:
    """Resolves the expected lockfile path for a given python version and extra."""
    lockfile_name = f"lockfile.{extra or 'core'}.txt"
    return pathlib.Path(rootdir) / python_version / lockfile_name


def resolve_coverage_datafile_path(
    python_version: str,
    extra: str | None = None,
) -> pathlib.Path:
    """Resolves the expected coverage data_file path for a given python version and extra."""
    coverage_datafile_name = f".coverage.{extra or 'core'}"
    return pathlib.Path(COVERAGE_DIR) / python_version / coverage_datafile_name


def resolve_junitxml_path(
    python_version: str,
    extra: str | None = None,
) -> pathlib.Path:
    """Resolves the output pytest junitxml reports path for a given python version and extra."""
    junitxml_report_name = f".junitxml.{extra or 'core'}.xml"
    return pathlib.Path(TEST_REPORTS_DIR) / python_version / junitxml_report_name


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def coverage_report(session: nox.Session) -> None:
    """Generate coverage reports.

    This session is usually triggered following a pytest coverage session generating
    the coverage data files to build the reports on. The directory containing
    the coverage data files should be provided as posarg.

    Examples:
        nox -s coverage_report-3.11 -- .coverage/3.11
    """

    session.install(*COVERAGE_TOOLS)

    # Combine coverage output data_files
    datafiles_dir = session.posargs[0]
    merged_data_file = f"{datafiles_dir}/.coverage"
    session.run(
        "coverage",
        "combine",
        "--data-file",
        merged_data_file,
        "--keep",
        datafiles_dir,
    )

    # Output coverage reports in same folder (for convenience)
    coverage_reports_dir = datafiles_dir
    session.run("coverage", "report", "--data-file", merged_data_file)
    session.run(
        "coverage",
        "html",
        "--data-file",
        merged_data_file,
        "-d",
        f"{coverage_reports_dir}/html",
    )
    session.run(
        "coverage",
        "xml",
        "--data-file",
        merged_data_file,
        "-o",
        f"{coverage_reports_dir}/coverage.xml",
    )
    session.run(
        "coverage",
        "json",
        "--data-file",
        merged_data_file,
        "-o",
        f"{coverage_reports_dir}/coverage.json",
    )

    session.notify(f"coverage_build_badge-{session.python}", [coverage_reports_dir])


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def coverage_build_badge(session: nox.Session) -> None:
    """Generate a coverage badge.

    This session is usually triggered following a pytest coverage session generating
    the coverage data files for which to build the badge. The directory containing
    the final .coverage data file should be provided as posarg.

    Examples:
        nox -s coverage_build_badge-3.11 -- .coverage/3.11
    """

    # coverage-badge only works from the same directory where the .coverage
    # data file is located.
    data_file_dir = session.posargs[0]
    session.chdir(data_file_dir)

    badge_filename = "coverage.svg"

    # cleanup old badge
    session.run("rm", "-rf", badge_filename, external=True)

    session.install(*COVERAGE_TOOLS)
    session.run("coverage-badge", "-o", badge_filename)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def dist_build(session: nox.Session) -> None:
    """Build distributions (sdist and wheel).

    The distribution packages are built using PyPA's `build`_ build frontend.
    This is the recommended way of building python packages, avoiding direct
    calls to the build backend. Legacy calls like ``$ python setup.py build``
    are now deprecated.

    .. _build:
            https://pypa-build.readthedocs.io/en/latest/

    Examples:
        nox -s dist_build-3.11
    """

    session.run("rm", "-rf", "dist", external=True)
    session.install(*BUILD_TOOLS)
    session.run("python", "-m", "build")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def docs_build(session: nox.Session) -> None:
    """Build sphinx documentation and API docs.

    Examples:
        nox -s docs_build-3.11
    """

    lockfile_path = resolve_lockfile_path(python_version=session.python)
    session.install(".[docs]", "--constraint", lockfile_path)

    # # Build API docs
    # apidoc_cmd = "sphinx-apidoc -f -o docs/source/pages/reference/api src/molflux --implicit-namespaces"
    # session.run(*apidoc_cmd.split(" "))

    # wipe artefacts from previous runs, in case there are any
    session.run("rm", "-rf", "docs/build/html", external=True)

    # -a -E flags make sure things are built from scratch
    build_cmd = "sphinx-build -a -E docs/source/ docs/build/html"

    # # Run doctests in the documentation
    # session.run(*build_cmd.split(" "), "-b", "doctest")

    # Build HTML pages
    session.run(*build_cmd.split(" "), "-b", "html")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def formatting_check(session: nox.Session) -> None:
    """Check codebase formatting.

    Examples:
        nox -s formatting_check-3.11
    """
    session.install(RUFF_VERSION)
    session.run("ruff", "check")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def formatting_fix(session: nox.Session) -> None:
    """Fix codebase formatting.

    Examples:
        nox -s formatting_fix-3.11
    """
    session.install(RUFF_VERSION)
    session.run("ruff", "format")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def linting_check(session: nox.Session) -> None:
    """Check codebase lint quality.

    Examples:
        nox -s linting_check-3.11
    """
    session.install(RUFF_VERSION)
    session.run("ruff", "check", *session.posargs, *SOURCE_FILES)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def linting_fix(session: nox.Session) -> None:
    """Fix codebase lint quality where possible.

    Examples:
        nox -s linting_fix-3.11
    """
    session.install(RUFF_VERSION)
    session.run("ruff", "check", "--fix", *session.posargs, *SOURCE_FILES)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def typing_check(session: nox.Session) -> None:
    """Check codebase type annotations.

    Examples:
        nox -s typing_check-3.11
        nox -s typing_check-3.11 -- --python-version 3.11
    """

    lockfile_path = resolve_lockfile_path(python_version=session.python)
    session.install(".[tests,typing]", "--constraint", lockfile_path)
    session.run("mypy", *session.posargs)


def generate_lockfile(
    session: nox.Session,
    extra: str | None,
    lockfile_path: pathlib.Path,
) -> None:
    """Generates a package dependencies' lockfile.

    Args:
        session: The nox.Session to use for the operation.
        extra: The name of an additional specific package extra to take into
            account when resolving dependencies. If not None, this will be used
            in addition to the usual 'docs' and 'tests' extras.
        lockfile_path: The path where to output the generated lockfile.
    """

    session.install("--upgrade", "pip", "pip-tools", "uv")

    package_extras = f"docs,tests,{extra}" if extra else "docs,tests"
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    session.run(
        "uv",
        "pip",
        "compile",
        *[f"--extra={x}" for x in package_extras.split(",")],
        "--strip-extras",
        "--no-emit-index-url",
        "pyproject.toml",
        "-o",
        str(lockfile_path),
        "--upgrade",
        env={"UV_CUSTOM_COMPILE_COMMAND": f"nox -s {session.name}"},
    )
    print(f"Lockfile generated at {str(lockfile_path)!r} ✨")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@nox.parametrize("extra", EXTRAS)
def dependencies_pin(session: nox.Session, extra: str | None) -> None:
    """Generate pinned dependencies lockfiles.

    Examples:
        nox -s dependencies_pin-3.11
        nox -s "dependencies_pin-3.11(extra=None)"
        nox -s "dependencies_pin-3.11(extra='sklearn')"
    """
    output_lockfile_path = resolve_lockfile_path(
        python_version=session.python,
        extra=extra,
    )
    generate_lockfile(session, extra=extra, lockfile_path=output_lockfile_path)


def run_tests(
    session: nox.Session,
    *args: str,
    extra: str | None,
    lockfile_path: pathlib.Path,
    notify: bool,
) -> None:
    """Runs tests.

    This includes running code snippets in our source code docstrings.

    Args:
        session: The nox.Session to use for the operation.
        extra: The name of the package extra for which to run tests. If None,
            only core tests will be collected.
        lockfile_path: The path to the lockfile to use for constraining the
            testing environment.
        notify: If True, coverage reporting sessions are queued up on success.
    """

    # Setup which files and tests to target
    if extra:
        # install test dependencies and extra dependencies
        package_extras = ",".join(["tests", extra])

        # all code requiring the extra is in its own dir
        doctests_target_dirs = [
            f"{dir}/{extra}"
            for dir in INVERTED_SUBMODULE_EXTRAS[extra]["doctests_target_dir_extras"]
        ]

        # all tests requiring the extra are in their own dir and
        tests_target_dirs = [
            f"{dir}/{extra}"
            for dir in INVERTED_SUBMODULE_EXTRAS[extra]["tests_target_dir_extras"]
        ]

    else:
        # only install test dependencies
        package_extras = "tests"

        # all code requiring extras in its own subdir - so parse everything
        # except those subdirs (see norecursedirs in pyproject.toml)
        doctests_target_dirs = INVERTED_SUBMODULE_EXTRAS[extra][
            "doctests_target_dir_core"
        ]

        # all tests not requiring extras are in their own dir
        tests_target_dirs = INVERTED_SUBMODULE_EXTRAS[extra]["tests_target_dir_core"]

    # Run tests
    install_cmd = f".[{package_extras}] --constraint {lockfile_path}"
    if extra == "pystan":  # handle pystan SyntaxError on uv backend
        install_cmd += " --compile"
    session.install(*install_cmd.split(" "))

    coverage_datafile_path = resolve_coverage_datafile_path(
        python_version=session.python,
        extra=extra,
    )
    junitxml_path = resolve_junitxml_path(python_version=session.python, extra=extra)
    session.run(
        "coverage",
        "run",
        f"--data-file={coverage_datafile_path}",
        "-m",
        "pytest",
        f"--junitxml={junitxml_path}",
        *args,
        *doctests_target_dirs,
        *tests_target_dirs,
    )

    # for parametrised runs, this will get run once at the end of all of them
    if notify:
        datafiles_dir = str(coverage_datafile_path.parent)
        session.notify(f"coverage_report-{session.python}", [datafiles_dir])


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@nox.parametrize("extra", EXTRAS)
def tests_run_latest(session: nox.Session, extra: str | None) -> None:
    """Run tests against latest available dependencies.

    Examples:
        nox -s tests_run_latest-3.11
        nox -s "tests_run_latest-3.11(extra=None)"
        nox -s "tests_run_latest-3.11(extra='sklearn')"
    """
    # Generate a scratch lock file with latest resolved dependencies
    #
    # we could have used session.create_tmp but that sets $TMPDIR which creates
    # problems with multiprocessing code: https://github.com/python/cpython/issues/93852
    with tempfile.TemporaryDirectory() as tmp:
        scratch_output_lockfile_path = resolve_lockfile_path(
            python_version=session.python,
            extra=extra,
            rootdir=tmp,
        )
        generate_lockfile(
            session,
            extra=extra,
            lockfile_path=scratch_output_lockfile_path,
        )

        run_tests(
            session,
            *session.posargs,
            extra=extra,
            lockfile_path=scratch_output_lockfile_path,
            notify=False,
        )


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@nox.parametrize("extra", EXTRAS)
def tests_run_pinned(session: nox.Session, extra: str | None) -> None:
    """Run tests against pinned dependencies.

    These should already be present. If not, they can be generated / updated
    by running the `dependencies_pin` session.

    Examples:
        nox -s tests_run_pinned-3.11
        nox -s "tests_run_pinned-3.11(extra=None)"
        nox -s "tests_run_pinned-3.11(extra='sklearn')"
    """
    expected_lockfile_path = resolve_lockfile_path(
        python_version=session.python,
        extra=extra,
    )
    run_tests(
        session,
        *session.posargs,
        extra=extra,
        lockfile_path=expected_lockfile_path,
        notify=True,
    )
