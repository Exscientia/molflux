#!/bin/bash

_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd "${_SCRIPT_DIR}" # cd to script directory

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

echo "Setting up virtualenv"

python -m venv .venv
.venv/bin/pip install --upgrade pip  # >= 21.3.1

PYTHON_VERSION=$(python -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))')
.venv/bin/pip install -e .[dev] --constraint pinned-versions/"$PYTHON_VERSION"/lockfile.core.txt --config-settings editable_mode=strict

popd # return to original directory
unset _SCRIPT_DIR
