#!/bin/bash

set -e

if [[ "$(uname)" == "Linux" ]]; then
    export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
fi

python3 -c "from PIL import Image"

python3 -bb -m pytest -v -x -W always --cov PIL --cov Tests --cov-report term Tests $REVERSE
