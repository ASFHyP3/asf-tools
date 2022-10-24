#!/bin/bash --login

set -e

conda activate hyp3-autorift

exec python -um /asf-tools/asf_tools/hand/produce_upload_hand.py "$@"


