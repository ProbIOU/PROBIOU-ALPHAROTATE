# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 500
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH * 20
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# Enables the dataaug
IMG_ROTATE = True
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# dataset
DATASET_NAME = 'MLT'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# train with PROBIOU loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 5.0
REG_LOSS_MODE = 9  # PROB IOU loss
VERSION = 'RetinaNet_MLT_PROBIOU_1x_20220128'

# # train with KL loss
# KL_TAU = 2.0
# KL_FUNC = 0   # 0: sqrt  1: log
# CLS_WEIGHT = 1.0
# REG_WEIGHT = 1.0
# REG_LOSS_MODE = 3  # PROB KL loss
# VERSION = 'RetinaNet_MLT_KL_1x_20211021'


# train with GWD loss
# GWD_TAU = 2.0  # The workaround parammeters
# GWD_FUNC = tf.sqrt
# CLS_WEIGHT = 1.0
# REG_WEIGHT = 1.0
# REG_LOSS_MODE = 2  # PROB GWD loss
# VERSION = 'RetinaNet_MLT_GWD_1x_20211021'


# train with L1 loss
# CLS_WEIGHT = 1.0
# REG_WEIGHT = 1.0
# REG_LOSS_MODE = 11  # PROB L1 loss
# VERSION = 'RetinaNet_MLT_L1_1x_20211021'


# post-processing
VIS_SCORE = 0.2
