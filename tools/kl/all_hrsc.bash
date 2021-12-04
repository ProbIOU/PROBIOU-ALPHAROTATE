#!/bin/bash
for i in 200000 #199992 #69996 79996 89996 99996 109996 119996 129996 139996 149996 159996 169996 179996 189996 199992
do 
    python3 test_hrsc2016.py --img_dir='/datasets/HRSC2016/Test/AllImages/' --test_annotation_path='/datasets/HRSC2016/Test/xmls/' --gpu='0,1,2,3' --checkpoint $i >  $i.txt
done
