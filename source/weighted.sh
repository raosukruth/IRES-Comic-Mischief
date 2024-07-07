#!/bin/bash 

cd /mnt/scratch/raosukru/IRES/IRES-Comic-Mischief
source bin/activate
cd /mnt/scratch/raosukru/IRES/IRES-Comic-Mischief/source

python ComicMischiefTest.py weighted pretrain > weighted_7_6_24.out 2>&1