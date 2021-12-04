# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
sys.path.append("../../")

from libs.models.detectors.r3det_kl import build_whole_network
from tools.test_mlt_base import TestMLT
from libs.configs import cfgs


class TestMLTKL(TestMLT):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        r3det_kl = build_whole_network.DetectionNetworkR3DetKL(cfgs=self.cfgs,
                                                    is_training=False)
        self.test_mlt(det_net=r3det_kl, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestMLTKL(cfgs)
    tester.eval()


