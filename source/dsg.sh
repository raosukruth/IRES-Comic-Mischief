#!/bin/bash 

cd /mnt/scratch/raosukru/IRES/IRES-Comic-Mischief
source bin/activate
cd /mnt/scratch/raosukru/IRES/IRES-Comic-Mischief/source

python ComicMischiefTest.py dsg pretrain > dsg_7_6_24.out 2>&1