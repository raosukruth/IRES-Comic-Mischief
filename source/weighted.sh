#!/bin/bash 

cd /home/srao22
source bin/activate
cd /home/srao22/project/IRES-Comic-Mischief-Copy/IRES-Comic-Mischief/source

nohup python -u ComicMischiefTest.py weighted pretrain > weighted_7_6_24.out 2>&1 &