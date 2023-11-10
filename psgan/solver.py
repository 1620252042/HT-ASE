#!/usr/bin/python
# -*- encoding: utf-8 -*-
import itertools
import os
import os.path as osp
import cv2
from clearml import Task, Logger
from torchvision.transforms import ToPILImage

from ops.SPL_Loss import GPLoss, CPLoss

pwd = osp.split(osp.realpath(__file__))[0]

import time
import datetime
from .encode_vgg import EncodeVGG
from .SCDis import Discriminator

import torch
from torch import nn
from torchvision.utils import save_image
import torch.nn.init as init
from torch.autograd import Variable
from ops.dischange_loss import DischangeLoss
from ops.loss_added import GANLoss
from ops.histogram_loss import HistogramLoss
import tools.plot as plot_fig
from . import net
from concern.track import Track
from tqdm import tqdm
from psgan.utils.helpers import FaceCropper
import numpy as np
# Task.init('lmx', '8PSGAN')


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    # feat_mean, feat_std=[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    normalized=(feat-feat_mean)/feat_std
    return normalized

class Solver(Track):
    def __init__(self, config, device="cpu", data_loader=None, inference=False):
        self.G = net.Generator()
        if inference:
            self.G.load_state_dict(torch.load(inference, map_location=torch.device(device)))
            self.G = self.G.to(device).eval()
            return

        self.start_time = time.time()
        self.checkpoint = config.MODEL.WEIGHTS
        self.log_path = config.LOG.LOG_PATH
        self.result_path = os.path.join(self.log_path, config.LOG.VIS_PATH)
        self.snapshot_path = os.path.join(self.log_path, config.LOG.SNAPSHOT_PATH)
        self.log_step = config.LOG.LOG_STEP
        self.vis_step = config.LOG.VIS_STEP
        if device == 'cuda':
            self.snapshot_step = config.LOG.SNAPSHOT_STEP // torch.cuda.device_count()
        else:
            self.snapshot_step = config.LOG.SNAPSHOT_STEP // 1

        # Data loader
        self.data_loader_train = data_loader
        self.img_size = config.DATA.IMG_SIZE

        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.num_epochs_decay = config.TRAINING.NUM_EPOCHS_DECAY
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.g_step = config.TRAINING.G_STEP
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2

        self.lambda_idt = config.LOSS.LAMBDA_IDT
        self.lambda_A = config.LOSS.LAMBDA_A
        self.lambda_B = config.LOSS.LAMBDA_B
        self.lambda_his_lip = config.LOSS.LAMBDA_HIS_LIP
        self.lambda_his_skin = config.LOSS.LAMBDA_HIS_SKIN
        self.lambda_his_eye = config.LOSS.LAMBDA_HIS_EYE
        self.lambda_vgg = config.LOSS.LAMBDA_VGG
        self.adv_weight = 1
        # Hyper-parameteres
        self.d_conv_dim = config.MODEL.D_CONV_DIM
        self.d_repeat_num = config.MODEL.D_REPEAT_NUM
        self.norm = config.MODEL.NORM

        self.device = device

        self.build_model()
        super(Solver, self).__init__()

    # For generator
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv2d') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def build_model(self):
        self.fc = FaceCropper(predictor_dir='/data/shape_predictor_68_face_landmarks.dat', )
        # self.G = net.Generator()
        # 构建两个判别器（二分类器） D_X, D_Y
        self.encode_vgg = EncodeVGG()
        self.disLA = Discriminator(input_nc=3, ndf=64, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=64, n_layers=5)

        # 初始化网络参数，apply为从nn.module继承
        self.G.apply(self.weights_init_xavier)
        self.disLA.apply(self.weights_init_xavier)
        self.disLB.apply(self.weights_init_xavier)

        # 从checkpoint文件中加载网络参数
        # self.load_checkpoint()

        self.criterionL1 = torch.nn.L1Loss()  # 循环一致性损失Cycle consistency
        self.criterionL2 = torch.nn.MSELoss()  # 感知损失Perceptual loss
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.criterionHis = HistogramLoss()  # 妆容损失makeup loss
        self.MSE_loss = torch.nn.MSELoss()
        self.dischange_loss = DischangeLoss()  # 妆容损失makeup loss


        # Optimizers 优化器，迭代优化生成器和判别器的参数
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.beta1, self.beta2),
                                            weight_decay=0.0001)
        self.d_A_optimizer = torch.optim.Adam(self.disLA.parameters(), lr=self.d_lr, betas=(self.beta1, self.beta2),
                                            weight_decay=0.0001)
        self.d_B_optimizer = torch.optim.Adam(self.disLB.parameters(), lr=self.d_lr, betas=(self.beta1, self.beta2),
                                            weight_decay=0.0001)

        # # Print networks
        # self.print_network(self.G, 'G')
        # self.print_network(self.D_A, 'D_A')
        # self.print_network(self.D_B, 'D_B')

        if torch.cuda.is_available():
            self.device = "cuda"
            if torch.cuda.device_count() > 1:
                self.G = nn.DataParallel(self.G)
                # self.disGA = nn.DataParallel(self.disGA)
                # self.disGB = nn.DataParallel(self.disGB)
                self.disLA = nn.DataParallel(self.disLA)
                self.disLB = nn.DataParallel(self.disLB)
                self.encode_vgg = nn.DataParallel(self.encode_vgg)
                self.criterionHis = nn.DataParallel(self.criterionHis)
                self.criterionGAN = nn.DataParallel(self.criterionGAN)
                self.criterionL1 = nn.DataParallel(self.criterionL1)
                self.criterionL2 = nn.DataParallel(self.criterionL2)
                self.criterionGAN = nn.DataParallel(self.criterionGAN)
                self.MSE_loss = nn.DataParallel(self.MSE_loss)

            self.G.cuda()
            self.encode_vgg.cuda()
            self.criterionHis.cuda()
            self.criterionGAN.cuda()
            self.criterionL1.cuda()
            self.criterionL2.cuda()
            # self.disGA.cuda()
            # self.disGB.cuda()
            self.disLA.cuda()
            self.disLB.cuda()
            self.MSE_loss.cuda()


    def load_checkpoint(self):
        G_path = os.path.join(self.checkpoint, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location=torch.device(self.device)))
            print('loaded trained generator {}..!'.format(G_path))

        disLA_path = os.path.join(self.checkpoint, 'disLA.pth')
        if os.path.exists(disLA_path):
            self.disLA.load_state_dict(torch.load(disLA_path, map_location=torch.device(self.device)))
            print('loaded trained discriminator disLA {}..!'.format(disLA_path))

        disLB_path = os.path.join(self.checkpoint, 'disLB.pth')
        if os.path.exists(disLB_path):
            self.disLB.load_state_dict(torch.load(disLB_path, map_location=torch.device(self.device)))
            print('loaded trained discriminator disLB {}..!'.format(disLB_path))

    def test(self, mask_c, mask_s, non_makeup, makeup):
        cur_prama = None
        with torch.no_grad():
           fake_A,_= self.G(mask_c, mask_s, non_makeup, makeup,phase="test")
        fake_A = fake_A.squeeze(0)

        # normalize
        min_, max_ = fake_A.min(), fake_A.max()
        fake_A.add_(-min_).div_(max_ - min_ + 1e-5)

        return ToPILImage()(fake_A.cpu())

    def train(self):
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)
        # Start with trained model if exists
        g_lr = self.g_lr
        d_lr = self.d_lr
        start = 0

        for self.e in range(start, self.num_epochs):  # epoch
            self.i = 0
            for input in tqdm(self.data_loader_train):
                self.i += 1
                source_input = input[0]
                reference_input = input[1]
                # for self.i, (source_input, reference_input) in tqdm(enumerate(self.data_loader_train)):  # batch
                # self.i += 1
                # (self.i), (source_input_m, reference_input_m) = enumerate(self.data_loader_train)

                if isinstance(source_input[1], dict) or isinstance(reference_input[1], dict):
                    print("No eyes!!")
                    continue
                    # image, mask

                non_makeup, makeup = source_input[0].to(self.device), reference_input[0].to(self.device)
                mask_c, mask_s = source_input[1].to(self.device), reference_input[1].to(self.device)
                dist_s, dist_r = source_input[2].to(self.device), reference_input[2].to(self.device)
                mask_dischange_eye_mouth_c, mask_dischange_eye_mouth_s = source_input[3].to(self.device), reference_input[3].to(self.device)
                mask_dischange_bachground_c, mask_dischange_bachground_s = source_input[4].to(self.device), reference_input[4].to(self.device)


                p_transfer, p_removel, p_rec_non_makeup, p_rec_makeup, p_cycle_makeup, p_cycle_non_makeup = self.G(
                    mask_c, mask_s, non_makeup, makeup)

                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B  判断是否是“真reference” y
                # Real
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(makeup)
                p_transfer_detach = Variable(p_transfer.data).detach()
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(p_transfer_detach)

                D_ad_loss_LA = self.MSE_loss(real_LA_logit,
                                             torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(
                    fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit,
                                                 torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(
                    fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))

                D_loss_A = self.adv_weight * ( D_ad_loss_LA + D_ad_cam_loss_LA)
                self.d_A_optimizer.zero_grad()
                D_loss_A.backward(retain_graph=False)
                self.d_A_optimizer.step()

                self.loss = {}
                self.loss['D-A-loss'] = D_loss_A.mean().item()

                # training D_B, D_B aims to distinguish class A 判断是否是“真source” x
                # Real
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(non_makeup)
                p_removel_detach = Variable(p_removel.data).detach()  # 看判别器判断是否是生成的素颜图像的能力
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(p_removel_detach.detach())

                D_ad_loss_LB = self.MSE_loss(real_LB_logit,
                                             torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(
                    fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit,
                                                 torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(
                    fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

                D_loss_B = self.adv_weight * ( D_ad_loss_LB + D_ad_cam_loss_LB)
                # Backward + Optimize
                # print("DB before", torch.cuda.memory_allocated() / 1024 / 1024)
                self.d_B_optimizer.zero_grad()
                D_loss_B.backward(retain_graph=False)
                self.d_B_optimizer.step()
                # print("DB after", torch.cuda.memory_allocated() / 1024 / 1024)
                # Logging
                self.loss['D-B-loss'] = D_loss_B.mean().item()

                # self.track("Discriminator backward")

                # ================== Train G ================== #
                if (self.i + 1) % self.g_step == 0:
                    # identity loss
                    assert self.lambda_idt > 0
                    # loss_idt
                    # print("idt_A:",p_rec_makeup.shape)
                    loss_idt_A = self.criterionL1(p_rec_makeup,
                                                  makeup) * self.lambda_A * self.lambda_idt  # 化妆的Identity loss
                    loss_idt_B = self.criterionL1(p_rec_non_makeup,
                                                  non_makeup) * self.lambda_B * self.lambda_idt  # 素颜的identity loss
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5

                    # GAN loss D_A(G)
                    # GAN loss D_B(G)
                    fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(p_transfer)
                    fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(p_removel)

                    G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
                    G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit,
                                                     torch.ones_like(fake_LA_cam_logit).to(self.device))
                    G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
                    G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit,
                                                     torch.ones_like(fake_LB_cam_logit).to(self.device))

                    g_A_loss_adv = self.adv_weight * (G_ad_loss_LA + G_ad_cam_loss_LA)
                    g_B_loss_adv = self.adv_weight * (G_ad_loss_LB + G_ad_cam_loss_LB)

                    # print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE", g_B_loss_adv)
                    # color_histogram loss
                    # 各局部颜色直方图损失  Makeup loss

                    input_data = (self.de_norm(non_makeup) * 255).squeeze()
                    target_data = (self.de_norm(makeup) * 255).squeeze()

                    target_match_lip = target_data * mask_s[:,0].squeeze(0)
                    target_match_skin = target_data * mask_s[:,1].squeeze(0)
                    target_match_eye = target_data * mask_s[:,2].squeeze(0)
                    target_match_makeup = target_match_lip + target_match_skin + target_match_eye

                    target_match_lip = input_data * mask_c[:,0].squeeze(0)
                    target_match_skin = input_data * mask_c[:,1].squeeze(0)
                    target_match_eye = input_data * mask_c[:,2].squeeze(0)
                    target_match_input = target_match_lip + target_match_skin + target_match_eye

                    # col=torch.gather(dist_s,dim=1,[[0],[1]])
                    z_transfer, _ = self.fc.warpFace(
                        target_match_input, target_match_makeup,
                        before_points=dist_s[:, :, [1, 0]], after_points=dist_r[:, :, [1, 0]],
                        # before_points=None, after_points=None,
                        use_poisson=True,
                        use_stasm=False,
                        additional_gaussian=False,
                        clone_method=cv2.NORMAL_CLONE,
                        use_tps=False,
                        extra_blending_for_extreme=True,
                        hue_threshold=1,
                        extra_blending_weight=0.15,
                        adjust_value=True
                    )

                    z_transfer = z_transfer.permute(2, 0, 1).unsqueeze(0)  # H,W,C-->C,H,W
                    # z_transfer = self.to_var(z_transfer, requires_grad=False)
                    z_removel, _ = self.fc.warpFace(
                        target_match_makeup, target_match_input,
                        before_points=dist_r[:, :, [1, 0]], after_points=dist_s[:, :, [1, 0]],
                        # before_points=None, after_points=None,
                        use_poisson=True,
                        use_stasm=False,
                        additional_gaussian=False,
                        clone_method=cv2.NORMAL_CLONE,
                        use_tps=False,
                        extra_blending_for_extreme=True,
                        hue_threshold=1,
                        extra_blending_weight=0.15,
                        adjust_value=True
                    )

                    z_removel = z_removel.permute(2, 0, 1).unsqueeze(0)  # H,W,C-->C,H,W

                    # if self.i % 1 == 0:
                    #     Logger.current_logger().report_image('p_transer', 'input_dataa', self.i,image=input_data.permute(1, 2,0).detach().cpu().numpy())
                    #     Logger.current_logger().report_image('p_transer', 'z_transferr', self.i,image=z_transfer.squeeze(0).permute(1, 2,0).detach().cpu().numpy())
                    #     Logger.current_logger().report_image('p_transer', 'target_dataa', self.i,image=target_data.permute(1, 2,0).detach().cpu().numpy())
                    # Logger.current_logger().report_image('p_transer', 'z_removal', self.i,
                    #                                      image=z_removel.squeeze(0).permute(1, 2,
                    #                                                                         0).detach().cpu().numpy())
                    # loss_G_CP = self.CPL(p_transfer, normal(z_transfer)) + self.CPL(p_removel, normal(z_removel))
                    # loss_G_GP = self.GPL(p_transfer, non_makeup) + self.GPL(p_removel, makeup)
                    # loss_G_SPL = loss_G_CP * 2 + loss_G_GP * 1

                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    g_A_lip_loss_his = self.criterionHis(p_transfer, z_transfer,makeup, mask_c[:, 0],
                                                         mask_s[:, 0])* self.lambda_his_lip
                    g_B_lip_loss_his = self.criterionHis(p_removel, z_removel,non_makeup, mask_s[:, 0],
                                                         mask_c[:, 0])* self.lambda_his_lip
                    g_A_loss_his += g_A_lip_loss_his  # 嘴唇色彩直方图的loss
                    g_B_loss_his += g_B_lip_loss_his

                    g_A_skin_loss_his = self.criterionHis(p_transfer, z_transfer,makeup, mask_c[:, 1],
                                                          mask_s[:, 1])* self.lambda_his_skin
                    g_B_skin_loss_his = self.criterionHis(p_removel, z_removel,non_makeup, mask_s[:, 1],
                                                          mask_c[:, 1])* self.lambda_his_skin
                    g_A_loss_his += g_A_skin_loss_his  # 皮肤色彩直方图的loss
                    g_B_loss_his += g_B_skin_loss_his

                    g_A_eye_loss_his = self.criterionHis(p_transfer, z_transfer,makeup, mask_c[:, 2],
                                                         mask_s[:, 2])* self.lambda_his_eye
                    g_B_eye_loss_his = self.criterionHis(p_removel, z_removel,non_makeup, mask_s[:, 2],
                                                         mask_c[:, 2])* self.lambda_his_eye
                    # # fixme mark loss
                    g_A_loss_his += g_A_eye_loss_his  # 眼部色彩直方图的loss
                    g_B_loss_his += g_B_eye_loss_his

                    dischange_c_bachground = self.dischange_loss(p_transfer, non_makeup,
                                                                 mask_dischange_bachground_c.squeeze(
                                                                     0)) * self.lambda_his_skin
                    dischange_s_bachground = self.dischange_loss(p_removel, makeup, mask_dischange_bachground_s.squeeze(
                        0)) * self.lambda_his_skin
                    dischange_c_eye = self.dischange_loss(p_transfer, non_makeup, mask_dischange_eye_mouth_c.squeeze(0))*0.3
                    dischange_s_eye = self.dischange_loss(p_removel, makeup, mask_dischange_eye_mouth_s.squeeze(0))*0.3
                    dischange_bachground = dischange_s_bachground + dischange_c_bachground
                    dischange_eye = dischange_c_eye + dischange_s_eye
                    g_loss_dischange = dischange_bachground + dischange_eye


                    # input_match_a=input_match_a_eye+input_match_a_lip+input_match_a_skin
                    # input_match_b = input_match_b_eye+ input_match_b_lip+input_match_b_skin
                    # if self.i % 1 == 0:
                    #     Logger.current_logger().report_image('p_transer', 'input_match_a', self.i,image=input_match_a.permute(1, 2,0).detach().cpu().numpy())

                    # cycle loss
                    # fake_A: fake_x/source #训练生成器 G的第一个参数是素颜图提供人脸身份信息的，第二个参数是提供妆容信息的
                    # 循环一致性损失，上面生成的化妆图和原素颜图再送进去应该生成卸妆图 p_cycle_makeup, p_cycle_non_makeup
                    g_loss_rec_A = self.criterionL1(p_cycle_makeup, makeup) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(p_cycle_non_makeup,non_makeup) * self.lambda_B

                    # vgg loss  感知损失是判断身份信息的loss
                    # Perceptual loss
                    vgg_makeup = self.encode_vgg(makeup)[-1]  # 参考图和生成的图像妆容一致
                    vgg_makeup = Variable(vgg_makeup.data).detach()
                    vgg_p_removal = self.encode_vgg(p_removel)
                    g_loss_A_vgg = self.criterionL2(vgg_p_removal[-1], vgg_makeup) * self.lambda_A * self.lambda_vgg

                    vgg_non_makeup = self.encode_vgg(non_makeup)[-1]  # 素颜图和生成的图像身份特征一致    #提取原素颜图的人脸身份信息
                    vgg_non_makeup = Variable(vgg_non_makeup.data).detach()
                    vgg_p_transfer = self.encode_vgg(p_transfer)  # 提取生成的化妆图的人脸身份信息
                    g_loss_B_vgg = self.criterionL2(vgg_p_transfer[-1],
                                                    vgg_non_makeup) * self.lambda_B * self.lambda_vgg

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg)*0.5

                    # Combined loss
                    g_loss = (g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his+g_loss_dischange).mean()


                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=False)
                    self.g_optimizer.step()
                    # self.track("Generator backward")

                    # Logging
                    self.loss['G-A-loss-adv'] = g_A_loss_adv.mean().item()  # GAN的生成对抗损失D(G)
                    self.loss['G-B-loss-adv'] = g_A_loss_adv.mean().item()  # GAN的生成对抗损失D(G)
                    # self.loss['GA_ad_loss'] = GA_ad_loss.mean().item()  # GAN的生成对抗损失D(G)
                    # self.loss['GB_ad_loss'] = GB_ad_loss.mean().item()  # GAN的生成对抗损失D(G)
                    # self.loss['LA_ad_loss'] = LA_ad_loss.mean().item()  # GAN的生成对抗损失D(G)
                    # self.loss['LB_ad_loss'] = LB_ad_loss.mean().item()  # GAN的生成对抗损失D(G)
                    self.loss['G-loss_rec_A'] = g_loss_rec_A.mean().item()  # 能不能算回原图
                    self.loss['G-loss_rec_B'] = g_loss_rec_B.mean().item()  # 能不能算回原图
                    self.loss['G-loss-idt'] = loss_idt.mean().item()  # 同一张图既做参考图又做素颜图出来的结果应该不变
                    self.loss['g_loss_rec_A + g_loss_rec_B'] = (g_loss_rec_A + g_loss_rec_B).mean().item()
                    self.loss['g_loss_A_vgg + g_loss_B_vgg'] = (g_loss_A_vgg + g_loss_B_vgg).mean().item()  # 感知损失

                    self.loss['G-A-loss-his'] = g_A_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['G-B-loss-his'] = g_B_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_A_eye_loss_his'] = g_A_eye_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_B_eye_loss_his'] = g_B_eye_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_A_lip_loss_his'] = g_A_lip_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_B_lip_loss_his'] = g_B_lip_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_A_skin_loss_his'] = g_A_skin_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_B_skin_loss_his'] = g_B_skin_loss_his.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_loss_dischange'] = g_loss_dischange.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_loss_dischange_bachground'] = dischange_bachground.mean().item()  # 直方图损失，的出来的图像色彩差距如何
                    self.loss['g_loss_dischange_eye'] = dischange_eye.mean().item()  # 直方图损失，的出来的图像色彩差距如何

                # Print out log info
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal()

                # plot the figures
                for key_now in self.loss.keys():
                    plot_fig.plot(key_now, self.loss[key_now])

                # save the images
                if self.i % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train(
                        [non_makeup, makeup, p_transfer, p_removel, p_cycle_non_makeup, p_cycle_makeup,
                         p_rec_non_makeup, p_rec_makeup, mask_c[:, :, 0], mask_s[:, :, 0]])

                # Save model checkpoints
                if (self.i) % self.snapshot_step == 0:
                    self.save_models()

                if (self.i % 100 == 99):
                    plot_fig.flush(self.log_path)

                plot_fig.tick()

            # Decay learning rate
            if (self.e + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_A_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.d_B_optimizer.param_groups:
            param_group['lr'] = d_lr

    def save_models(self):
        if not osp.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        torch.save(
            self.G.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))

        torch.save(
            self.disLA.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_disLA.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.disLB.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_disLB.pth'.format(self.e + 1, self.i + 1)))

    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = osp.join(self.result_path, mode)
        if not osp.exists(result_path_train):
            os.makedirs(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e + 1, self.num_epochs, self.i + 1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)
