import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmcv.runner import BaseModule
# from .grad_rev import GradientReversal
from torch.autograd import Variable

from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            # alpha.retain_grad()
            # print("self.alpha grad:",alpha.grad)
            grad_input = - alpha*grad_output
        # print("grad_ouput",grad_output)
        # print("grad_input",grad_input)
        return grad_input, None

class GradientReversalM(nn.Module):
    def __init__(self):
        super().__init__()
        # self.alpha = torch.tensor(alpha, requires_grad=True)
        # self.alpha = alpha

    def forward(self, x, alpha):
        return GradientReversal.apply(x, alpha)

class lamda_scheduler():
    def __init__(self, r):
        self.r = r
    def update(self, p):
        lamda = 2 / (1 + np.exp(-self.r * p)) - 1 # from 0 to 1
        return torch.tensor(lamda.astype(float), requires_grad=False)
    

@HEADS.register_module()
class DAHead(BaseModule):
    def __init__(self,useCTB=False,loss=None):
        super(DAHead, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                  padding=0, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1,
                               padding=0, bias=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1,
                               padding=0, bias=True)
        self.fc = nn.Linear(100, 1, bias=True)
        # self.flatten = nn.Flatten()
        # self.cls_logits =  nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d([2, 2])
        self.init_weights()
        self.loss = build_loss(loss)
        # self.alpha1 = Variable(torch.cuda.FloatTensor([0.001]), requires_grad=True)# 1
        self.alpha2 = lamda_scheduler(10) # 2
        self.grl = GradientReversalM()
        self.total = 0
        self.count = 0.0
        self.img_idx = 0
        self.total_img = 6488 * 2
        original_array = np.arange(self.total_img)
        self.normalized_array = (original_array - np.min(original_array)) / (np.max(original_array) - np.min(original_array))
        self.GradCAM = False
        self.acc = False
        self.Alpha = torch.tensor(1.0, requires_grad=False)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.useCTB = useCTB
    
    def forward(self, in_da_feat, img_metas):
        
        if self.training:
            if self.img_idx >= self.total_img:
                self.img_idx = self.total_img - 1   
                # print("p:",self.normalized_array[self.img_idx])    
                self.Alpha = self.alpha2.update(self.normalized_array[self.img_idx])
                self.img_idx = self.img_idx + 1
            else:
                # print("p:",self.normalized_array[self.img_idx])
                self.Alpha = self.alpha2.update(self.normalized_array[self.img_idx])
                self.img_idx = self.img_idx + 1
        # print("Alpha", self.Alpha)
        # for x in mlvl:
        #     x = self.grl(x, self.Alpha)
        #     x = F.relu(self.conv1(x))
        #     x = self.conv2(x)
        #     x = F.relu(x)  
        #     x = self.conv3(x)
        #     while x.shape[2] != mlvl[-1].shape[2] and x.shape[3] != mlvl[-1].shape[3]:
        #         x = self.pool(x)
        #     # print(x.shape)
        #     x = self.flatten(x)
        #     cat =+ x 
        # cat = cat / 6
        # cat = torch.mean(cat, 1, True)
        # prob_cat = F.sigmoid(cat)
        loss_ret = 0.0
        in_da_feat_ = in_da_feat[-1]
        for i, feat in enumerate(in_da_feat_):
            x = self.grl(feat, self.Alpha)
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            x = F.relu(x)  
            x = self.conv3(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            # x = self.cls_logits(x)
            if img_metas[i]['filename'] is not None:
                if '_target' in img_metas[i]['filename']:
                    label = torch.full(x.shape, 1, dtype=torch.float, device=x.device)  
                else:
                    label = torch.full(x.shape, 0, dtype=torch.float, device=x.device) 
            else:
                if img_metas[i]['ori_shape'][1] == 2048:
                    label = torch.full(x.shape, 1, dtype=torch.float, device=x.device)  
                else:
                    label = torch.full(x.shape, 0, dtype=torch.float, device=x.device) 
            loss_ret = loss_ret + self.loss_fn(x, label) 

            # print("prob", F.sigmoid(x))
        loss_ret = loss_ret / len(in_da_feat_)

        ret =  {'da_loss': loss_ret}

        if self.GradCAM:
            return [loss_ret]
        elif self.acc:
            # for i, l in enumerate(prob):
            #     self.total = self.total + 1
            #     if torch.abs(torch.sub(l, labels[i])) < 0.2:
            #         self.count = self.count + 1.0
            prob = F.sigmoid(x)
            self.total = self.total + 1
            # print(torch.mean(torch.abs(torch.sub(prob, label))))
            if torch.mean(torch.abs(torch.sub(prob, label))) < 0.2:
                self.count = self.count + 1.0
            if self.total == 10:
                acc = (self.count / self.total) * 100.0
                self.count = 0.0
                self.total = 0
                # print("Alpha", self.Alpha)
                return acc, ret
            else:
                return None, ret