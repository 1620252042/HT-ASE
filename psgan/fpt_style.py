# ---------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import DataParallel  # or your customized DataParallel module
# from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback
# ‘_a’ means all

from .self_trans import SelfTrans
from .rendering_trans import RenderTrans
from .grounding_trans import GroundTrans



class FPT_style(nn.Module):
    def __init__(self, feature_dim, upsample_method='bilinear'):    #feature 256
        super(FPT_style, self).__init__()
        self.feature_dim = feature_dim  #256
        assert upsample_method in ['nearest', 'bilinear']

        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method,
                                 align_corners=False if upsample_method == 'bilinear' else None)

        self.fpn_upsample = interpolate


        # self.st_p3 = SelfTrans(n_head=1, n_mix=4, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        # self.st_p2 = SelfTrans(n_head=1, n_mix=4, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        # self.st_p1 = SelfTrans(n_head=1, n_mix=4, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)

        self.gt_p2_p3 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,bn_layer=True)
        # self.gt_p1_p2 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
        #                             bn_layer=True)
        # self.gt_p1_p3 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
        #                             bn_layer=True)

        # self.rt_p3_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        # self.rt_p3_p1 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p2_p1 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)

        # self.fpn_p4_1x1 = nn.Conv2d(256, feature_dim, 1)    #64*64,256
        self.fpn_p3_1x1 = nn.Conv2d(256, feature_dim, 1)    #64*64,256
        self.fpn_p2_1x1 = nn.Conv2d(128, feature_dim, 1)    #128*128,128
        self.fpn_p1_1x1 = nn.Conv2d(64, feature_dim, 1)     #256*256,64

        # self.fpt_p3 = nn.Conv2d(feature_dim * 4, feature_dim, 3, padding=1)
        self.fpt_p2 = nn.Conv2d(feature_dim * 4, feature_dim, 3, padding=1)
        # self.fpt_p1 = nn.Conv2d(feature_dim * 4, feature_dim, 3, padding=1)
        # self.str_conv3x3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False)


        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res1, res2, res3):
        fpn_p1_1 = self.fpn_p1_1x1(res1)  # 128*128,128
        # fpn_p1_1_conv = self.str_conv3x3(fpn_p1_1)    #256*256,64
        fpn_p2_1 = self.fpn_p2_1x1(res2)    #128*128,128
        fpn_p3_1 = self.fpn_p3_1x1(res3)    #64*64,256

        # fpt_p4_out = torch.cat((self.st_p4(fpn_p4_1), self.rt_p4_p3(fpn_p4_1, fpn_p3_1),
        #                         self.rt_p4_p1(fpn_p4_1, fpn_p1_1), self.gt_p4_p6(fpn_p4_1, fpn_p6_1), fpn_p4_1), 1)

        # fpt_p3_out = torch.cat((self.st_p3(fpn_p3_1), self.rt_p3_p2(fpn_p3_1, fpn_p2_1),
        #                         self.rt_p3_p1(fpn_p3_1, fpn_p1_1_conv), fpn_p3_1), 1)
        # fpt_p3 = self.fpt_p3(fpt_p3_out)

        fpt_p2_out = torch.cat((fpn_p2_1, self.rt_p2_p1(fpn_p2_1, fpn_p1_1),
                                self.gt_p2_p3(fpn_p2_1, fpn_p3_1), fpn_p2_1), 1)
        fpt_p2 = self.fpt_p2(fpt_p2_out)

        # fpt_p1_out = torch.cat((self.st_p1(fpn_p1_1_conv), self.gt_p1_p2(fpn_p1_1_conv, fpn_p2_1),
        #                         self.gt_p1_p3(fpn_p1_1_conv, fpn_p3_1), fpn_p1_1_conv), 1)
        # fpt_p1 = self.fpt_p1(fpt_p1_out)


        return fpt_p2
