#!/bin/bash
for i in 200004 #69996 79996 89996 99996 109996 119996 129996 139996 149996 159996 169996 179996 189996 199992:q

do
    python3 test_icdar2015.py --test_dir='/datasets/ICDAR2015/Test/Images/' --gpus='0,1,2,3' --checkpoint $i
    python3 eval_icdar/script.py  >  $i.txt
done
