#!/bin/bash --login

string=`conda env list |grep "*"`

echo "conda env $string"

echo "calcualte the HAND based on the  $@"

mkdir -p /home/conda/output

cd /home/conda/output

string=`which python`

echo "python: $string"

echo "python /asf-tools/asf_tools/hand/produce_hand_jhk.py $@"

python /asf-tools/asf_tools/hand/produce_hand_jhk.py "$@"

echo "completed ... "

