import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy
from mmdet.utils import get_root_logger

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class MSCAM(nn.Module):

    def __init__(self, channels=64, r=4):
        super(MSCAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei
        return xo

def setup_seed(seed):
    random.seed(seed)                          
    numpy.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

class SAPNet(nn.Module):

    def __init__(self, *, num_anchors, in_channels, embedding_kernel_size=3,
            embedding_norm=False, embedding_dropout=True, func_name='cross_entropy',
            pool_type='avg', window_strides=None,
            window_sizes=(3, 9, 15, -1)
        ):
        super().__init__()
        num_anchors = num_anchors * 9
        self.in_channels = in_channels
        self.embedding_kernel_size = embedding_kernel_size
        self.embedding_norm = embedding_norm
        self.embedding_dropout = embedding_dropout
        self.num_windows = len(window_sizes)
        self.num_anchors = num_anchors
        self.window_sizes = window_sizes
        if window_strides is None:
            self.window_strides = [None] * len(window_sizes)
        else:
            assert len(window_strides) == len(window_sizes), 'window_strides and window_sizes should has same len'
            self.window_strides = window_strides

        if pool_type == 'avg':
            channel_multiply = 1
            pool_func = F.avg_pool2d
        elif pool_type == 'max':
            channel_multiply = 1
            pool_func = F.max_pool2d
        else:
            raise ValueError('{}, only support avg and max pooling'.format(pool_type))
        self.pool_type = pool_type
        self.pool_func = pool_func

        if func_name == 'cross_entropy':
            num_domain_classes = 2
            loss_func = F.cross_entropy
        else:
            raise ValueError
        self.func_name = func_name
        self.loss_func = F.cross_entropy
        self.num_domain_classes = num_domain_classes

        NormModule = nn.BatchNorm2d if embedding_norm else nn.Identity
        DropoutModule = nn.Dropout if embedding_dropout else nn.Identity

        self.grl = GradientScalarLayer(-1.0)
        padding = (embedding_kernel_size - 1) // 2
        bias = not embedding_norm
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(in_channels),
            nn.ReLU(True),
            DropoutModule(),

            nn.Conv2d(in_channels, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
            DropoutModule(),

            nn.Conv2d(256, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
            DropoutModule(),
        )

        self.shared_semantic = nn.Sequential(
            nn.Conv2d(in_channels + self.num_anchors, in_channels, kernel_size=embedding_kernel_size, stride=1, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding),
            nn.ReLU(True),
        )



        # self.mscam = MSCAM(channels=in_channels+num_anchors, r=1)
        self.semantic_list = nn.ModuleList()

        self.inter_channels = 128
        for _ in range(self.num_windows):
            self.semantic_list += [
                nn.Sequential(
                    nn.Conv2d(256, 128, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 1, 1),
                )
            ]

        self.fc = nn.Sequential(
            nn.Conv2d(256 * channel_multiply, 128, 1, bias=False),
            NormModule(128),
            nn.ReLU(inplace=True),
        )

        self.split_fc = nn.Sequential(
            nn.Conv2d(128, self.num_windows * 256 * channel_multiply, 1, bias=False),
        )

        self.predictor = nn.Linear(256 * channel_multiply, num_domain_classes)

        self.conv1 = nn.Linear(256 * channel_multiply, 128)
        self.conv2 = nn.Linear(128, 64)
        self.conv3 = nn.Linear(64, 2)
        self.fcc = nn.Linear(100, 1, bias=True)


        self.logger_n = get_root_logger()


    def forward(self, feature, rpn_logits, input_domain):
        '''
        Args:
            feature: feature comes from backbone [N, c, h, w]
            rpn_logits: feature comes from rpn (anchors), list[tensor], [N, c, h, w] but in different size 
        Returns: dict<str, tensor>, domain loss name and loss tensor
        '''
        setup_seed(42)
        feature = self.grl(feature)
        rpn_logits_ = []
        for r in rpn_logits:
            r = self.grl(r)
            if feature.shape != r.shape:
                r = F.interpolate(r, size=(feature.size(2), feature.size(3)), mode='bilinear', align_corners=True)
            rpn_logits_.append(r)
        semantic_map = torch.cat([feature, *rpn_logits_], dim=1)
        # semantic_map = self.mscam(semantic_map) # MSCAM
        semantic_map = self.shared_semantic(semantic_map)
        feature = self.embedding(feature) # feature embedding, input is features coming from backbone
        N, C, H, W = feature.shape

        pyramid_features = []
        for i, k in enumerate(self.window_sizes):
            if k == -1:
                x = self.pool_func(feature, kernel_size=(H, W))
            elif k == 1:
                x = feature
            else:
                stride = self.window_strides[i]
                if stride is None:
                    stride = 1  # default
                x = self.pool_func(feature, kernel_size=k, stride=stride) # spatial pyramid pooling
            _, _, h, w = x.shape
            semantic_map_per_level = F.interpolate(semantic_map, size=(h, w), mode='bilinear', align_corners=True)# down sampling at spatial attention part
            domain_logits = self.semantic_list[i](semantic_map_per_level) # spatial attention weight, it will re-weight features

            w_spatial = domain_logits.view(N, -1)
            w_spatial = F.softmax(w_spatial, dim=1)


            w_spatial = w_spatial.view(N, 1, h, w)
            x = torch.sum(x * w_spatial, dim=(2, 3), keepdim=True) # spatial attention re-weighting
            pyramid_features.append(x)

        fuse = sum(pyramid_features)  # [N, 256, 1, 1]

        # channel-wise attention part
        merge = self.fc(fuse)  # [N, 128, 1, 1] channel-wise attention fc layer, to generate channel-wise weights

        split = self.split_fc(merge)  # [N, num_windows * 256, 1, 1]
        split = split.view(N, self.num_windows, -1, 1, 1)

        w = F.softmax(split, dim=1) # channel-wise attention weights
        w = torch.unbind(w, dim=1)  # List[N, 256, 1, 1]

        pyramid_features = list(map(lambda x, y: x * y, pyramid_features, w)) # channel-wise attention re-weighting
        final_features = sum(pyramid_features)
        del pyramid_features, w, split, merge, fuse, feature, rpn_logits
        final_features = final_features.view(N, -1) # semantic vector
        # final_features = F.relu(self.conv1(final_features))
        # final_features = self.conv2(final_features)
        # final_features = F.relu(final_features)  
        # logits = self.conv3(final_features)
        # logits = final_features.view(x.size(0), -1)
        # logits = self.fc(final_features)
        logits = self.predictor(final_features)

        if input_domain[-1] == 'source':
            # self.logger_n.info(f"source_logits: {logits}")
            label = torch.ones(logits.size(0), dtype=torch.long, device=logits.device) ########
            # self.logger_n.info(f"label: {label}")
            domain_loss = self.loss_func(logits, label) ########
            with torch.no_grad():
                s_acc = (torch.softmax(logits, dim=1).argmax(dim=1) == 1).float().mean() ########
                # self.logger_n.info(f"SAP s_acc: {s_acc}")
            # self.logger_n.info(f"loss_sap_source_domain: {domain_loss}")
            return {'loss_sap_source_domain': domain_loss}
        elif input_domain[-1] == 'target':
            # self.logger_n.info(f"target_logits: {logits}")
            label = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device) ########
            # self.logger_n.info(f"label: {label}")
            domain_loss = self.loss_func(logits, label) ########
            with torch.no_grad():
                t_acc = (torch.softmax(logits, dim=1).argmax(dim=1) == 0).float().mean() ########
                # self.logger_n.info(f"SAP t_acc: {t_acc}")
            # self.logger_n.info(f"loss_sap_target_domain: {domain_loss}")
            return {'loss_sap_target_domain': domain_loss}


    @classmethod
    def from_config(cls, cfg):
        return {
            'num_anchors': cfg.MODEL.DA_HEAD.NUM_ANCHOR_IN_IMG,
            'in_channels': cfg.MODEL.DA_HEAD.IN_CHANNELS,
            'embedding_kernel_size': cfg.MODEL.DA_HEAD.EMBEDDING_KERNEL_SIZE,
            'embedding_norm': cfg.MODEL.DA_HEAD.EMBEDDING_NORM,
            'embedding_dropout': cfg.MODEL.DA_HEAD.EMBEDDING_DROPOUT,
            'func_name': cfg.MODEL.DA_HEAD.FUNC_NAME,
            'pool_type': cfg.MODEL.DA_HEAD.POOL_TYPE,
            'window_strides': cfg.MODEL.DA_HEAD.WINDOW_STRIDES,
            'window_sizes': cfg.MODEL.DA_HEAD.WINDOW_SIZES,
        }

    def spatial_attention_mask(self, feature, rpn_logits):
        '''
        Args:
            feature: feature comes from backbone [N, c, h, w]
            rpn_logits: feature comes from rpn (anchors), list[tensor], [N, c, h, w] but in different size 
        Returns: List[torch.Tensor], spatial attention masks
        '''
        feature = self.grl(feature)
        rpn_logits_ = []
        for r in rpn_logits:
            r = self.grl(r)
            if feature.shape != r.shape:
                r = F.interpolate(r, size=(feature.size(2), feature.size(3)), mode='bilinear', align_corners=True)
            rpn_logits_.append(r)
        semantic_map = torch.cat([feature, *rpn_logits_], dim=1)
        # semantic_map = self.mscam(semantic_map) # MSCAM
        semantic_map = self.shared_semantic(semantic_map)
        feature = self.embedding(feature) # feature embedding, input is features coming from backbone
        N, C, H, W = feature.shape

        spatial_mask_list = []
        for i, k in enumerate(self.window_sizes):
            if k == -1:
                x = self.pool_func(feature, kernel_size=(H, W))
            elif k == 1:
                x = feature
            else:
                stride = self.window_strides[i]
                if stride is None:
                    stride = 1  # default
                x = self.pool_func(feature, kernel_size=k, stride=stride) # spatial pyramid pooling
            _, _, h, w = x.shape
            semantic_map_per_level = F.interpolate(semantic_map, size=(h, w), mode='bilinear', align_corners=True)# down sampling at spatial attention part
            domain_logits = self.semantic_list[i](semantic_map_per_level) # spatial attention weight, it will re-weight features

            w_spatial = domain_logits.view(N, -1)
            w_spatial = F.softmax(w_spatial, dim=1)

            w_spatial = w_spatial.view(N, 1, h, w)
            spatial_mask_list.append(w_spatial.view(N, h, w))
        return spatial_mask_list

    def logit_vector(self, feature, rpn_logits):
        '''
        Args:
            feature: feature comes from backbone [N, c, h, w]
            rpn_logits: feature comes from rpn (anchors), list[tensor], [N, c, h, w] but in different size 
        Returns: torch.Tensor, semantic vector, [N, c, 1, 1]
        '''

        feature = self.grl(feature)
        rpn_logits_ = []
        for r in rpn_logits:
            r = self.grl(r)
            if feature.shape != r.shape:
                r = F.interpolate(r, size=(feature.size(2), feature.size(3)), mode='bilinear', align_corners=True)
            rpn_logits_.append(r)
        semantic_map = torch.cat([feature, *rpn_logits_], dim=1)
        # semantic_map = self.mscam(semantic_map) # MSCAM
        semantic_map = self.shared_semantic(semantic_map)
        feature = self.embedding(feature) # feature embedding, input is features coming from backbone
        N, C, H, W = feature.shape

        pyramid_features = []
        for i, k in enumerate(self.window_sizes):
            if k == -1:
                x = self.pool_func(feature, kernel_size=(H, W))
            elif k == 1:
                x = feature
            else:
                stride = self.window_strides[i]
                if stride is None:
                    stride = 1  # default
                x = self.pool_func(feature, kernel_size=k, stride=stride) # spatial pyramid pooling
            _, _, h, w = x.shape
            semantic_map_per_level = F.interpolate(semantic_map, size=(h, w), mode='bilinear', align_corners=True)# down sampling at spatial attention part
            domain_logits = self.semantic_list[i](semantic_map_per_level) # spatial attention weight, it will re-weight features

            w_spatial = domain_logits.view(N, -1)
            w_spatial = F.softmax(w_spatial, dim=1)


            w_spatial = w_spatial.view(N, 1, h, w)
            x = torch.sum(x * w_spatial, dim=(2, 3), keepdim=True) # spatial attention re-weighting
            pyramid_features.append(x)

        fuse = sum(pyramid_features)  # [N, 256, 1, 1]

        # channel-wise attention part
        merge = self.fc(fuse)  # [N, 128, 1, 1] channel-wise attention fc layer, to generate channel-wise weights

        split = self.split_fc(merge)  # [N, num_windows * 256, 1, 1]
        split = split.view(N, self.num_windows, -1, 1, 1)

        w = F.softmax(split, dim=1) # channel-wise attention weights
        w = torch.unbind(w, dim=1)  # List[N, 256, 1, 1]

        pyramid_features = list(map(lambda x, y: x * y, pyramid_features, w)) # channel-wise attention re-weighting
        final_features = sum(pyramid_features)
        del pyramid_features, w, split, merge, fuse, feature, rpn_logits
        final_features = final_features.view(N, -1) # semantic vector
        logits = self.predictor(final_features)
        return logits