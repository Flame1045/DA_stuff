# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_wA.py"
# WORKDIR1="outputs/custom_sim2city_unsupervised_lamda_scheduler511_da_head_D"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR1 --deterministic --adapter --adapter_choose adapter da_head

##################

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_wA.py"
# WORKDIR2="outputs/custom_sim2city_unsupervised_lamda_scheduler511_da_head_scalar_D"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR2 --deterministic --adapter --adapter_choose adapter da_head scalar

##################

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
# WORKDIR3="outputs/custom_sim2city_unsupervised_lamda_scheduler01_da_head_only_scalar_new256D_after_transformer"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR3 --deterministic

##################

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_wA.py"
# WORKDIR4="outputs/custom_sim2city_unsupervised_lamda_scheduler01_da_head_adapter_only_scalar_new256D"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR4 --load-from "outputs/custom_sim2city_unsupervised_lamda_scheduler01_da_head_all_scalar_new256D/iter_28000.pth" \
# --deterministic --adapter --adapter_choose adapter da_head scalar


# #####20240118#####source only#####

# CONFIG="projects/configs/co_dino/custom_sim2city_sourceonly.py"
# WORKDIR='outputs/ONEB1_sim2city_unsupervised_sourceonly'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
# --load-from 'pretrained/co_dino_5scale_9encoder_lsj_r50_3x_coco.pth'


# #####20240118#####SMF+SAF#####

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
# WORKDIR="outputs/ONEB1_sim2city_unsupervised_woA"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
# --load-from 'outputs/ONEB1_sim2city_unsupervised_sourceonly/latest.pth'


# ###################################################################

# #####20240119#####source only#####

# CONFIG="projects/configs/co_dino/custom_sim2city_sourceonly.py"
# WORKDIR='outputs/ONEB2_sim2city_unsupervised_sourceonly'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
# --load-from 'pretrained/co_dino_5scale_9encoder_lsj_r50_3x_coco.pth'


# #####20240119#####SMF+SAF#####

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
# WORKDIR="outputs/ONEB2_sim2city_unsupervised_woA"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
# --load-from 'outputs/ONEB2_sim2city_unsupervised_sourceonly/latest.pth'


#####TEST#####
# python3  tools/test.py "outputs/ONEB1_sim2city_unsupervised_sourceonly/custom_sim2city_sourceonly.py" \
#          "outputs/ONEB1_sim2city_unsupervised_sourceonly/latest.pth" \
#          --eval bbox
            ###############################################################################
         #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.705
            ###############################################################################
#####TEST#####

#####20240122#####Train(All)+SMF+SAF+Dcls#####

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
# WORKDIR="outputs/ONEB3_sim2city_unsupervised_woA"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --resume-from $WORKDIR/latest.pth --seed 134084244 \


#####20240124#####Train(SMF+SAF+Dcls+ADAPTER)#####

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_wA.py"
# WORKDIR="outputs/ONEB3_sim2city_unsupervised_wA"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter da_head neck \
#     --load-from 'outputs/ONEB1_sim2city_unsupervised_sourceonly/latest.pth' --seed 134084244 \
    
#####20240131#####Train(SMF+SAF+Dcls)#####

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
# WORKDIR="outputs/ONEB4_sim2city_unsupervised_wA"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --da_head --load-from 'outputs/ONEB1_sim2city_unsupervised_sourceonly/latest.pth' --seed 134084244 \


# #####20240206#####source only#####

# CONFIG="projects/configs/co_dino/custom_sim2city_base.py"
# WORKDIR='outputs/SSO_NEW'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#    --load-from 'pretrained/co_deformable_detr_r50_1x_coco.pth'

# #####20240206#####New#####

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 \

#####TEST#####BASE
# python3  tools/test.py "outputs/SSO_NEW/custom_sim2city_base.py" \
#          "outputs/SSO_NEW/latest.pth" \
#          --eval bbox
            ##############################################################################
         #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.641
            ##############################################################################
#####TEST#####

#####TEST#####BASE_WA
# python3  tools/test.py "outputs/BASE_WA/custom_sim2city_unsupervised_base_wA.py" \
#          "outputs/BASE_WA/iter_80000.pth" \
#          --eval bbox
            ##############################################################################
         #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.649
            ##############################################################################
#####TEST#####

#####20240206#####Train(VIS GRADCAM)#####
# uncomment /media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/mmdet/models/detectors/base.py 
# line 446 ~ 466
#########################################

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_vis.py"
# WORKDIR="outputs/GradCAM"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --grad_cam --load-from 'outputs/BASE_WA/iter_80000.pth' --seed 134084244 \

# #####20240223#####WithPGT#####
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_PGT'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic --pseudo_label_flag \
#     --adapter --adapter_choose adapter da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 \

# #####20240227#####WOPGT + NATTEN#####
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_WOPGT_NATTEN'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter da_head spatial_fusion_module global_fusion_module saf_module na2d scalar \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 \

   
# #####20240227#####ADAPTER#####
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_NO_DCLS'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 \

# #####20240227#####ADAPTERTest4#####
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test4_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.0000002" optimizer.weight_decay="0.0001" \


# #####20240227#####ADAPTERTest5#####
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test5_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.000002" optimizer.weight_decay="0.00001" \

# #####20240227#####ADAPTERTest6#####
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test6_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.00002" optimizer.weight_decay="0.000001" \

#####20240227#####ADAPTERTest7#####    65.4 mAP
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test7_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \

######20240227#####ADAPTERTest3#####
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test3'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 \


#####20240305#####BASE#####

# CONFIG="projects/configs/co_dino/custom_sim2city_base.py"
# WORKDIR='outputs/BASE-Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#    --load-from 'pretrained/co_deformable_detr_r50_1x_coco.pth'


#####20240305#####ADAPTERTest10########## Dcls 0to1 loss 

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test10_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module  \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####20240305#####ADAPTERTest11########## Dcls 0to1 loss 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test11_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####20240305#####ADAPTERTest12########## Dcls 0to1 loss 

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_NAT.py"
# WORKDIR='outputs/BASE_WA_Test12_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module na2d \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \


#####20240305#####ADAPTERTest13########## Dcls 0to1 loss 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_NAT.py"
# WORKDIR='outputs/BASE_WA_Test13_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module na2d global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \


#####ADAPTERTest11########## Adapter in DEC, CTB, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test11_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####ADAPTERTest11VIS########## Adapter in DEC, CTB, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_vis.py"
# WORKDIR='outputs/DEBUG'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --grad_cam \
#     --load-from "outputs/BASE_WA_Test11_Batch8/iter_30500.pth" --seed 134084244 


#####ADAPTERTest14########## CTB, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woA.py"
# WORKDIR='outputs/BASE_WA_Test14_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \


#####ADAPTERTest15########## Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woAnCTB.py"
# WORKDIR='outputs/BASE_WA_Test15_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \


#####ADAPTERTest16########## Adapter in ENC, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_inENC_woCTB.py"
# WORKDIR='outputs/BASE_WA_Test16_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####ADAPTERTest17########## Adapter in ENC, CTB, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_inENC.py"
# WORKDIR='outputs/BASE_WA_Test17_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####ADAPTERTest18########## Adapter in DEC, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTB.py"
# WORKDIR='outputs/BASE_WA_Test18_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####ADAPTERTest19########## Adapter in DEC, CTB, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test19_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \


#####20240311#####source only BASE_RES50#####

# CONFIG="projects/configs/co_dino/custom_sim2city_base.py"
# WORKDIR='outputs/BASE_RES50'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic

######ADAPTERTest22####BASE_RES50###### CTB, Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
# WORKDIR='outputs/BASE_WA_Test22_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####ADAPTERTest20####BASE_RES50###### Dcls 0to1 loss 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woAnCTB.py"
# WORKDIR='outputs/BASE_WA_Test20_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

#####ADAPTERTest21####BASE_RES50###### CTB, Dcls 0to1 loss
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woA.py"
# WORKDIR='outputs/BASE_WA_Test21_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

######ADAPTERTest23####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTB.py"
# WORKDIR='outputs/BASE_WA_Test23_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

######ADAPTERTest33####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test33_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \

# ######ADAPTERTest32####BASE_RES50###### CTB, Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wAV2.py"
# WORKDIR='outputs/BASE_WA_Test32_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \


# adapter_natten_V2 

######ADAPTERTest34####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test34_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
#     model.query_head.decoder.operation_order="('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'adapter_natten_V2a7x7', 'norm')" \

######ADAPTERTest35####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test35_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
#     model.query_head.decoder.operation_order="('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'adapter_natten_V2a13x13', 'norm')" \


######ADAPTERTest36####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test36_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
#     model.query_head.decoder.operation_order="('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'adapter_natten_V2a3x3', 'norm')" \

######ADAPTERTest37####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test37_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
#     model.query_head.decoder.operation_order="('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'adapter_natten_V2a7x7R', 'norm')" \

######ADAPTERTest38####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test38_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
#     model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'adapter_natten_V2a7x7', 'norm')" \
    

######ADAPTERTest39####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test39_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
#     model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'adapter_natten_V2a3x3', 'norm')" \

######ADAPTERTest40####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test40_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
#     model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'adapter_natten_V2a5x5R', 'norm')" \

######ADAPTERTest43####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
WORKDIR='outputs/BASE_WA_Test43_Batch8'

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
    --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
    --load-from "outputs/BASE_RES50/latest.pth" --seed 134084244 \
    --cfg-options optimizer.lr="0.002" optimizer.weight_decay="0.0000001" \
    model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_res_adapter', 'norm','ffn', 'adapter_natten_V2a7x7', 'norm')" \