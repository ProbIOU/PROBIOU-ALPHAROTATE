# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import shutil
import os
import random
import math


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


divide_rate = 0.7351

root_path = '/datasets/UCAS-AOD'

image_path = root_path + '/images'
xml_path = root_path + '/label-xml'

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

random.shuffle(image_name)

train_image = image_name[:int(math.ceil(len(image_name)) * divide_rate)]
test_image = image_name[int(math.ceil(len(image_name)) * divide_rate):]

image_output_train = os.path.join(root_path, 'Train/Images')
mkdir(image_output_train)
image_output_test = os.path.join(root_path, 'Test/Images')
mkdir(image_output_test)

xml_train = os.path.join(root_path, 'Train/Annotations')
mkdir(xml_train)
xml_test = os.path.join(root_path, 'Test/Annotations')
mkdir(xml_test)


count = 0
for i in train_image:
    shutil.copy(os.path.join(image_path, i + '.png'), image_output_train)
    shutil.copy(os.path.join(xml_path, i + '.xml'), xml_train)
    if count % 100 == 0:
        print("process step {}".format(count))
    count += 1

for i in test_image:
    shutil.copy(os.path.join(image_path, i + '.png'), image_output_test)
    shutil.copy(os.path.join(xml_path, i + '.xml'), xml_test)
    if count % 100 == 0:
        print("process step {}".format(count))
    count += 1
