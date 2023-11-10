import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from clearml import Task, Logger
from ops.histogram_matching import histogram_matching



class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def forward(self, input_data, target_data,makeup_data, mask_src, mask_tar):
        index_tmp = mask_src.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_tar.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]

        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data=target_data.squeeze()
        makeup_data = (self.de_norm(makeup_data) * 255).squeeze()


        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_src
        makeup_masked = makeup_data * mask_tar

        # target_masked = self.to_var(target_masked, requires_grad=False)

        input_match = histogram_matching(
            target_masked, makeup_masked,
            [x_A_index, y_A_index, x_B_index, y_B_index])
        input_match = self.to_var(input_match, requires_grad=False)

        # Logger.current_logger().report_image('p_transer', 'target_masked', 1,
        #                                      image=input_match.permute(1, 2,0).detach().cpu().numpy())
        loss = F.l1_loss(input_masked, input_match)

        # input_match = histogram_matching(
        #     target_masked, makeup_masked,
        #     [x_A_index, y_A_index, x_B_index, y_B_index])
        # input_match = self.to_var(input_match, requires_grad=False)
        #
        # Logger.current_logger().report_image('p_transer', 'input_match', 1,
        #                                      image=input_match.permute(1, 2,0).detach().cpu().numpy())
        # loss = F.l1_loss(input_masked, input_match)

        return loss
