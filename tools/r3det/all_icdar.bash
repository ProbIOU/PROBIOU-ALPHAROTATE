#!/bin/bash
for i in 199996 #70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 199996
do
    python3 test_icdar2015.py --test_dir='/datasets/ICDAR2015/Test/Images/' --gpus='0,1,2,3' --checkpoint $i
    python3 eval_icdar/script.py  >  $i.txt
done
