#!/bin/bash --login

echo "calculate the acc based on the  $@"

mkdir -p /home/conda/output

cd /home/conda/output

python /asf-tools/asf_tools/hand/acc_static.py "$@"

echo "completed ... "
