#!/bin/bash --login
set -e
conda activate asf-tools
exec make_hand "$@"