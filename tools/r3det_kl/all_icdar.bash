#!/bin/bash
for i in 199996 # 70000 80000 90004 100004 110004 120004 130004 140004 150004 160004 170004 180008 190008 #199996
do
    python3 test_icdar2015.py --test_dir='/datasets/ICDAR2015/Test/Images/' --gpus='0,1,2,3' --checkpoint $i
    python3 eval_icdar/script.py  >  $i.txt
done
