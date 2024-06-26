# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmdet.core.visualization import imshow_det_bboxes
from PIL import Image
import datetime

def generate_timestamp():
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date as a string (you can customize the format as needed)
    date_str = current_datetime.strftime("%Y%m%d_%H%M%S")

    return date_str

def Grad_CamDA(model, img, img_metas=None, gt_bboxes=None, gt_labels=None, idx=None, target_layers = None, iiter = None):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # model = resnet50(pretrained=True)
    target_layers = [target_layers] 
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

    targets = [BinaryClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, 
                        img_metas=img_metas,  #############
                        gt_bboxes=gt_bboxes,  #############
                        gt_labels=gt_labels,  #############
                        targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[idx, :]
    # img = np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0))
    rgbimg = mmcv.imread(img_metas[idx]['filename']).astype(np.uint8)
    rgbimg = mmcv.bgr2rgb(rgbimg)
    rgbimg = mmcv.imresize(rgbimg, (1024, 1024), return_scale=False)
    rgbimg = (rgbimg-np.min(rgbimg))/(np.max(rgbimg)-np.min(rgbimg))
    visualization = show_cam_on_image(rgbimg, grayscale_cam, use_rgb=True) 
    image = Image.fromarray(visualization, mode='RGB')
    time = generate_timestamp()
    image.save('Grad_Cam/' + 'DAhead_' + str(iiter) + '_' + time + '_'+ img_metas[idx]['filename'].split('/')[-1])

def Grad_CamRPN(model, img, img_metas=None, gt_bboxes=None, gt_labels=None, idx=None, target_layers = None, iiter = None):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # model = resnet50(pretrained=True)
    target_layers = [target_layers] 
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

    targets = [BinaryClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, 
                        img_metas=img_metas,  #############
                        gt_bboxes=gt_bboxes,  #############
                        gt_labels=gt_labels,  #############
                        targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[idx, :]
    # img = np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0))
    rgbimg = mmcv.imread(img_metas[idx]['filename']).astype(np.uint8)
    rgbimg = mmcv.bgr2rgb(rgbimg)
    rgbimg = mmcv.imresize(rgbimg, (1024, 1024), return_scale=False)
    rgbimg = (rgbimg-np.min(rgbimg))/(np.max(rgbimg)-np.min(rgbimg))
    visualization = show_cam_on_image(rgbimg, grayscale_cam, use_rgb=True) 
    image = Image.fromarray(visualization, mode='RGB')
    image.save('Grad_Cam/' + 'RPNhead_' + str(iiter) + '_' + img_metas[idx]['filename'].split('/')[-1])

def Grad_CamAdapter(model, img, img_metas=None, gt_bboxes=None, gt_labels=None, idx=None, target_layers = None, iiter = None):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # model = resnet50(pretrained=True)
    target_layers = [target_layers] 
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

    targets = [BinaryClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, 
                        img_metas=img_metas,  #############
                        gt_bboxes=gt_bboxes,  #############
                        gt_labels=gt_labels,  #############
                        targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[idx, :]
    # img = np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0))
    rgbimg = mmcv.imread(img_metas[idx]['filename']).astype(np.uint8)
    rgbimg = mmcv.bgr2rgb(rgbimg)
    rgbimg = mmcv.imresize(rgbimg, (1024, 1024), return_scale=False)
    rgbimg = (rgbimg-np.min(rgbimg))/(np.max(rgbimg)-np.min(rgbimg))
    visualization = show_cam_on_image(rgbimg, grayscale_cam, use_rgb=True) 
    image = Image.fromarray(visualization, mode='RGB')
    image.save('Grad_Cam/' + 'Adapter_' + str(iiter) + '_' + img_metas[idx]['filename'].split('/')[-1])

def Grad_CamROI(model, img, img_metas=None, gt_bboxes=None, gt_labels=None, idx=None, target_layers = None, iiter = None):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # model = resnet50(pretrained=True)
    target_layers = [target_layers] 
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

    targets = [BinaryClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, 
                        img_metas=img_metas,  #############
                        gt_bboxes=gt_bboxes,  #############
                        gt_labels=gt_labels,  #############
                        targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[idx, :]
    # img = np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0))
    rgbimg = mmcv.imread(img_metas[idx]['filename']).astype(np.uint8)
    rgbimg = mmcv.bgr2rgb(rgbimg)
    rgbimg = mmcv.imresize(rgbimg, (1024, 1024), return_scale=False)
    rgbimg = (rgbimg-np.min(rgbimg))/(np.max(rgbimg)-np.min(rgbimg))
    visualization = show_cam_on_image(rgbimg, grayscale_cam, use_rgb=True) 
    image = Image.fromarray(visualization, mode='RGB')
    image.save('Grad_Cam/' + 'ROIhead_' + str(iiter) + '_'+ img_metas[idx]['filename'].split('/')[-1])

def Grad_CamBBOX(model, img, img_metas=None, gt_bboxes=None, gt_labels=None, idx=None, target_layers = None, iiter = None):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # model = resnet50(pretrained=True)
    target_layers = [target_layers] 
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

    targets = [BinaryClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, 
                        img_metas=img_metas,  #############
                        gt_bboxes=gt_bboxes,  #############
                        gt_labels=gt_labels,  #############
                        targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[idx, :]
    # img = np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0))
    rgbimg = mmcv.imread(img_metas[idx]['filename']).astype(np.uint8)
    rgbimg = mmcv.bgr2rgb(rgbimg)
    rgbimg = mmcv.imresize(rgbimg, (1024, 1024), return_scale=False)
    rgbimg = (rgbimg-np.min(rgbimg))/(np.max(rgbimg)-np.min(rgbimg))
    visualization = show_cam_on_image(rgbimg, grayscale_cam, use_rgb=True) 
    image = Image.fromarray(visualization, mode='RGB')
    image.save('Grad_Cam/' + 'BBOXhead_' + str(iiter) + '_' + img_metas[idx]['filename'].split('/')[-1])
    


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.num = np.random.randint(0, 41070, 1000)
        self.iiter = 1

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        # if not isinstance(imgs, list):
        #     imgs = [imgs]
        # if not isinstance(img_metas, list):
        #     img_metas = [img_metas]
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        self.iiter = self.iiter + 1
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        # if self.iiter in self.num:
        #     if self.da_head is not None:
        #         self.da_head.GradCAM = True
        #         for idx in range(1):
        #             print(data['img_metas'][idx]['filename'] + "da_head Visaulizing...")
        #             Grad_CamDA(self,data['img'], data['img_metas'], data['gt_bboxes'], data['gt_labels'], idx, self.da_head.conv3, self.iiter)
        #         self.da_head.GradCAM = False

        # if self.iiter in self.num:
        # self.rpn_head.GradCAM = True
        # for idx in range(1):
        #     print(data['img_metas'][idx]['filename'] + " rpn_head Visaulizing...")
        #     Grad_CamRPN(self,data['img'], data['img_metas'], data['gt_bboxes'], data['gt_labels'], idx, self.rpn_head.rpn_conv, self.iiter)
        # self.rpn_head.GradCAM = False

        # self.roi_head.GradCAM = True
        # for idx in range(1):
        #     print(data['img_metas'][idx]['filename'] + "query_head.transformer.encoder.layers.3.adapter Visaulizing...")
        #     Grad_CamAdapter(self,data['img'], data['img_metas'], data['gt_bboxes'], data['gt_labels'], idx, self.query_head.transformer.encoder.layers[4].adapter[0], self.iiter)
        # self.roi_head.GradCAM = False
       

        # if self.iiter in self.num:
        # self.roi_head.GradCAM = True
        # for idx in range(1):
        #     print(data['img_metas'][idx]['filename'] + " roi_head Visaulizing...")
        #     Grad_CamROI(self,data['img'], data['img_metas'], data['gt_bboxes'], data['gt_labels'], idx, self.roi_head[0].bbox_head.fc_cls, self.iiter)
        # self.roi_head.GradCAM = False

        # if self.iiter in self.num:
        #     self.bbox_head.GradCAM = True
        #     for idx in range(1):
        #         print(data['img_metas'][idx]['filename'] + " bbox_head Visaulizing...")
        #         Grad_CamBBOX(self,data['img'], data['img_metas'], data['gt_bboxes'], data['gt_labels'], idx, self.bbox_head[0].cls_convs[0].conv, self.iiter)
        #     self.bbox_head.GradCAM = False
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')
