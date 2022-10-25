#!/bin/bash --login

set -e

conda activate asf-tools

exec python -u /asf-tools/asf_tools/hand/produce_upload_hand.py "$@"
