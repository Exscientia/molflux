#!/bin/bash

_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd "${_SCRIPT_DIR}" # cd to script directory

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

echo "Setting up virtualenv"

# Make sure to bypass a possible $PIP_REQUIRE_VIRTUALENV=true set by users
# This is because pip is unable to detect that it is inside conda's virtual environment - and would throw an error
env PIP_REQUIRE_VIRTUALENV=false conda env create -f environment.yaml --prefix .venv

popd # return to original directory
unset _SCRIPT_DIR
