#!/usr/bin/bash

export PYTHONPATH=./TaskHead:./Binary/source:./Binary
echo "Arguments are" $@
python -m TaskHead.taskheads taskheads.py $@