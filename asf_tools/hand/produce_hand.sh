#!/bin/bash --login

echo "calcualte the HAND based on the  $@"

mkdir -p /home/conda/output

cd /home/conda/output

python /asf-tools/asf_tools/hand/produce_hand.py "$@"

echo "completed ... "
