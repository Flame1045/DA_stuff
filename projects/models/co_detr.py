import warnings

import torch
import torch.nn as nn

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector
from mmdet.utils import get_root_logger

class SpatialWiseFusion(nn.Module):
    def __init__(self, input_channels):
        super(SpatialWiseFusion, self).__init__()

        # Normalization layer
        self.norm_layer = nn.BatchNorm2d(input_channels, affine=False)

        # Learnable weight map
        self.weight_map = nn.Parameter(torch.ones(1, input_channels, 1, 1))

        # Learnable bias map
        self.bias_map = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

        # Activation function
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Apply normalization
        normalized_x = self.norm_layer(x)

        # Apply re-weighting and re-biasing
        weighted_x = normalized_x * self.weight_map + self.bias_map

        # Apply activation function
        activated_x = self.activation(weighted_x)

        return activated_x

class ChannelWiseFusion(nn.Module):
    def __init__(self, input_channels):
        super(ChannelWiseFusion, self).__init__()

        # Global Average Pooling (GAP)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Learnable weight vector
        self.weight_vector = nn.Parameter(torch.ones(1, input_channels, 1, 1))

        # Learnable bias vector
        self.bias_vector = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply Global Average Pooling (GAP)
        pooled_x = self.global_avg_pooling(x)

        # Reshape to match the weight and bias vectors
        pooled_x = pooled_x.view(pooled_x.size(0), pooled_x.size(1), 1, 1)

        # Apply re-weighting and re-biasing
        weighted_x = pooled_x * self.weight_vector + self.bias_vector

        # Apply sigmoid activation function to limit the output in the range of 0 to 1
        output = self.sigmoid(weighted_x)

        return output



class ScaleAggregationFusion(nn.Module):
    def __init__(self, input_channels, num_scales):
        super(ScaleAggregationFusion, self).__init__()

        self.num_scales = num_scales
        # Global Average Pooling (GAP) for each scale
        self.gap_layers = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_scales)])

        # Fully connected layer to obtain scale-weight vectors A
        self.fc_layer = nn.Linear(input_channels, num_scales * input_channels)

        # Sigmoid activation to ensure scale weights are between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        # Calculate channel-wise vectors u for each scale
        scale_vectors = [gap_layer(feature) for gap_layer, feature in zip(self.gap_layers, feature_list)]

        # Merge channel-wise vectors using element-wise addition
        merged_vector = torch.stack(scale_vectors, dim=1).sum(dim=1)

        # Flatten the merged vector
        flattened_vector = merged_vector.view(merged_vector.size(0), -1)

        # Obtain scale-weight vectors using a fully connected layer
        scale_weights = self.fc_layer(flattened_vector)

        # Reshape the output to create a list of tensors
        # scale_weights_list = torch.chunk(scale_weights, chunks=self.num_scales, dim=1)

        # Reshape each tensor in the list to [batch_size, input_channels, 1, 1]
        scale_weights_list = [scale_weights[:, i * input_tensor.size(1):(i + 1) * input_tensor.size(1)].view(input_tensor.size(0), -1, 1, 1)
                              for i, input_tensor in enumerate(feature_list)]


        # # Apply sigmoid activation to ensure scale weights are between 0 and 1
        # scale_weights = self.sigmoid(scale_weights)

        # Multiply each feature by its corresponding scale weight and sum them
        mul_feature = [feature * scale_weights_list[i] for i, feature in enumerate(feature_list)]
        # fused_feature = torch.sum(torch.stack(mul_feature, dim=1), dim=1)

        return mul_feature, scale_weights


@DETECTORS.register_module()
class CoDETR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 da_head=None,
                 rpn_head=None,
                 roi_head=[None],
                 bbox_head=[None],
                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',
                 eval_index=0):
        super(CoDETR, self).__init__(init_cfg)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        # Module for evaluation, ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module
        # Module index for evaluation
        self.eval_index = eval_index
        self.backbone = build_backbone(backbone)
        self.logger_n = get_root_logger()
        # self.model2222 = self
        head_idx = 0

        if neck is not None:
            self.neck = build_neck(neck)

        if da_head is not None:
            self.da_head = build_head(da_head)
        else:
            self.da_head = None

        if query_head is not None:
            query_head.update(train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = build_head(query_head)
            self.query_head.init_weights()
            head_idx += 1

        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (train_cfg is not None and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i+head_idx].rcnn if (train_cfg and train_cfg[i+head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i+head_idx].rcnn)
                self.roi_head.append(build_head(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(train_cfg=train_cfg[i+head_idx+len(self.roi_head)] if (train_cfg and train_cfg[i+head_idx+len(self.roi_head)] is not None) else None)
                bbox_head[i].update(test_cfg=test_cfg[i+head_idx+len(self.roi_head)])
                self.bbox_head.append(build_head(bbox_head[i]))  
                self.bbox_head[-1].init_weights() 

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bbox_head.GradCAM = False
        self.rpn_head.GradCAM = False
        self.roi_head.GradCAM = False
        self.spatial_fusion_module = SpatialWiseFusion(input_channels=256//2).to("cuda:0")
        self.global_fusion_module = ChannelWiseFusion(input_channels=256//2).to("cuda:0")
        self.saf_module = ScaleAggregationFusion(input_channels=256, num_scales=5).to("cuda:0")

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head)>0))

    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.da_head is not None:
            self.da_head.acc = True

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        CNN_feat = list(x)

        losses = dict()
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses
        # gt_bboxes_ignore = []
        Pseudo_label_flag = True
        gt_bboxes_ignore_Flag = False
        for b, img_meta in enumerate(img_metas): #
            if "leftImg8bit" in img_meta["filename"]:
                gt_bboxes_ignore_Flag = True
                # print("gt_bboxes_ignore:",gt_bboxes_ignore)
            else:
                gt_bboxes_ignore_Flag = False

        # if img_metas[0]["ori_filename"][-4:] != img_metas[1]["ori_filename"][-4:]:
        #     assert print("~~~~~~~~",img_metas[0]["ori_filename"], img_metas[1]["ori_filename"])
        # DETR encoder and decoder forward

        if Pseudo_label_flag and gt_bboxes_ignore_Flag:
            dir = "/home/ee4012/Eric/new/Vary-toy/Vary-master/filter_GT/"
            gt_bboxes = []
            gt_labels = []
            for img_m in img_metas:
                filename = img_m['filename'].split('/')[-1].replace('.png','.txt')
                filename = dir + filename
                with open(filename, "r") as file:
                    lines = file.readlines()
                # Extract values after the word "car" from each line
                pgtbbox = []
                pgtlable = []
                for line in lines:
                    if "car" in line:
                        pgtbbox.append(list(map(float, line.strip().split()[1:])))
                        pgtlable.extend([0])
                # Convert the list of values to a tensor with size [2, 4]
                pgtbbox = torch.tensor(pgtbbox).to("cuda:0")
                gt_bboxes.append(pgtbbox)
                pgtlable = torch.tensor(pgtlable).to("cuda:0")
                gt_labels.append(pgtlable)
                if not pgtlable.numel()==0:
                    gt_bboxes_ignore_Flag = False

        #         print("filename:", filename)
        # print("gt_bboxes:", gt_bboxes)
        # print("gt_labels:", gt_labels)
        
        if self.with_query_head:
            bbox_losses, x = self.query_head.forward_train(x, img_metas, gt_bboxes,
                                                           gt_labels, gt_bboxes_ignore)
            if gt_bboxes_ignore_Flag:
                _ = bbox_losses
            else:
                losses.update(bbox_losses)
            TRNSF_feat = x

        if self.da_head is not None:
            # CT Blender
            # Split-Merge Fusion
            # Initialize lists to store cnn_chunks
            concatenated_hierarchical = []
            for i in range(0,4):
                Trn = TRNSF_feat[i]
                Cnn = CNN_feat[i]

                num_chunks = 2
                # Check if the input_tensor can be evenly split
                assert Trn.size(1) % num_chunks == 0, f"Cannot evenly split tensor along dimension 1 into {num_chunks} chunks."
                Trn_chunks = torch.chunk(Trn, num_chunks, dim=1) # T split channels (K groups)
                # Check if the input_tensor can be evenly split
                assert Cnn.size(1) % num_chunks == 0, f"Cannot evenly split tensor along dimension 1 into {num_chunks} chunks."
                Cnn_chunks = torch.chunk(Cnn, num_chunks, dim=1) # C split channels (K groups)

                # Initialize lists to store cnn_chunks
                concatenated_cnn_chunks = []
                for j in range(0,num_chunks):
                    # each layer of C and T
                    trn_chunks = Trn_chunks[j].to("cuda:0")
                    cnn_chunks = Cnn_chunks[j].to("cuda:0")

                    # spatial(5)
                    self.spatial_output_tensor = self.spatial_fusion_module(trn_chunks)

                    # global(6)
                    self.global_output_tensor = self.global_fusion_module(trn_chunks)
                
                    # re-weight by (5) and (6)
                    cnn_chunks = cnn_chunks * self.spatial_output_tensor
                    cnn_chunks = cnn_chunks * self.global_output_tensor

                    # shuffle K times


                    # Append the re-weighted cnn_chunks to the list
                    concatenated_cnn_chunks.append(cnn_chunks)

                # Concatenate the cnn_chunks outside the loop
                concatenated_cnn_chunks = torch.cat(concatenated_cnn_chunks, dim=1)
                concatenated_hierarchical.append(concatenated_cnn_chunks)

            # Forward pass
            fused_feature, scale_weights = self.saf_module(concatenated_hierarchical)
                
            in_da_feat = fused_feature  

            if self.da_head.GradCAM == False and self.rpn_head.GradCAM == False and self.bbox_head.GradCAM == False and self.roi_head.GradCAM == False:
                if self.da_head is not None:
                    acc, da_loss = self.da_head(in_da_feat, img_metas)
                    da_loss = upd_loss(da_loss, idx=0)
                    losses.update(da_loss)
                    if acc is not None:
                        self.logger_n.info(f"acc: {acc}")
                        self.logger_n.info(f"acc: {acc}")
                        self.logger_n.info(f"acc: {acc}")
                    # return losses

        # RPN forward and loss
        if self.with_rpn and not gt_bboxes_ignore_Flag:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                              self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if not gt_bboxes_ignore_Flag:
            positive_coords = []
            for i in range(len(self.roi_head)):
                roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
                if self.with_pos_coord:
                    positive_coords.append(roi_losses.pop('pos_coords'))
                else: 
                    if 'pos_coords' in roi_losses.keys():
                        tmp = roi_losses.pop('pos_coords')     
                roi_losses = upd_loss(roi_losses, idx=i)
                losses.update(roi_losses)
                
            for i in range(len(self.bbox_head)):
                bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore)
                if self.with_pos_coord:
                    pos_coords = bbox_losses.pop('pos_coords')
                    positive_coords.append(pos_coords)
                else:
                    if 'pos_coords' in bbox_losses.keys():
                        tmp = bbox_losses.pop('pos_coords')          
                bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
                losses.update(bbox_losses)

            if self.with_pos_coord and len(positive_coords)>0:
                for i in range(len(positive_coords)):
                    bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                                gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                    bbox_losses = upd_loss(bbox_losses, idx=i)
                    losses.update(bbox_losses)  

            query_list = self.simple_test_query_head(img, img_metas)           

        if self.bbox_head.GradCAM or self.rpn_head.GradCAM or self.roi_head.GradCAM and gt_bboxes_ignore_Flag:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                              self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            
            positive_coords = []
            for i in range(len(self.roi_head)):
                roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
                if self.with_pos_coord:
                    positive_coords.append(roi_losses.pop('pos_coords'))
                else: 
                    if 'pos_coords' in roi_losses.keys():
                        tmp = roi_losses.pop('pos_coords')     
                roi_losses = upd_loss(roi_losses, idx=i)
                losses.update(roi_losses)
                
            for i in range(len(self.bbox_head)):
                bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore)
                if self.with_pos_coord:
                    pos_coords = bbox_losses.pop('pos_coords')
                    positive_coords.append(pos_coords)
                else:
                    if 'pos_coords' in bbox_losses.keys():
                        tmp = bbox_losses.pop('pos_coords')          
                bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
                losses.update(bbox_losses)

            if self.with_pos_coord and len(positive_coords)>0:
                for i in range(len(positive_coords)):
                    bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                                gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                    bbox_losses = upd_loss(bbox_losses, idx=i)
                    losses.update(bbox_losses)  

            query_list = self.simple_test_query_head(img, img_metas)

        if self.da_head is not None and self.da_head.GradCAM == True:

            da_loss = self.da_head(in_da_feat, img_metas)
            return da_loss

        if self.rpn_head.GradCAM == True:

            rpn_loss = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            return rpn_loss[0]['loss_rpn_cls']
        
        if self.roi_head.GradCAM == True:
            
            for i in range(len(self.roi_head)):
                roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
                if self.with_pos_coord:
                    positive_coords.append(roi_losses.pop('pos_coords'))
                else: 
                    if 'pos_coords' in roi_losses.keys():
                        tmp = roi_losses.pop('pos_coords')   
                total_loss_tensor = torch.tensor([0.], device='cuda:0')  
                for key, value in losses.items():
                    if 'd4.loss_bbox' in key:
                        if isinstance(value, list):
                            value = sum(value)  # Convert scalar tensor to 1-dimensional tensor
                        total_loss_tensor += value
                # print(losses)
                # print(total_loss_tensor)
                # print("losses[loss_bbox0]",losses["loss_bbox0"])
                return total_loss_tensor
            
        if self.bbox_head.GradCAM == True:
            for i in range(len(self.bbox_head)):
                bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore)
                if self.with_pos_coord:
                    pos_coords = bbox_losses.pop('pos_coords')
                    positive_coords.append(pos_coords)
                else:
                    if 'pos_coords' in bbox_losses.keys():
                        tmp = bbox_losses.pop('pos_coords')          
                return bbox_losses['loss_cls']

        return losses


    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-2]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[self.eval_index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        results_list = self.query_head.simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-2]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self, 
                    img, 
                    img_metas, 
                    proposals=None, 
                    rescale=False,
                    **kwargs):
        if self.da_head is not None:
            self.da_head.acc = False
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']
        if self.with_bbox and self.eval_module=='one-stage':
            return self.simple_test_query_head(img, img_metas, proposals, rescale)
        if self.with_roi_head and self.eval_module=='two-stage':
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        if self.da_head is not None:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
            if not self.with_attn_mask: # remove attn mask for LSJ
                for i in range(len(img_metas)):
                    input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                    img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]
            x = self.extract_feat(img, img_metas)
            acc = self.da_head(x, img_metas)
            if acc is not None:
                self.logger_n.info(f"acc: {acc}")
                self.logger_n.info(f"acc: {acc}")
                self.logger_n.info(f"acc: {acc}")
        return self.simple_test_query_head(img, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.query_head, 'aug_test'), \
            f'{self.query_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.query_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.query_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.query_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
    
def Grad_Cam(model, img):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # model = resnet50(pretrained=True)
    target_layers = [model.da_head.conv3]
    input_tensor = img
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [ClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True) 