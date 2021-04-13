#!/bin/bash --login
set -e
conda activate asf-tools
exec calculate_hand "$@"