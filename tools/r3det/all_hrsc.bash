#!/bin/bash
for i in 199996 #60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 199996 
do
    python3 test_hrsc2016.py --img_dir='/datasets/HRSC2016/Test/AllImages/' --test_annotation_path='/datasets/HRSC2016/Test/xmls/' --gpu='0,1,2,3' --checkpoint $i >  $i.txt
done

