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
# /media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/projects/models/co_detr.py line 300 Pseudo_label_flag = True ############
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA.py"
WORKDIR='outputs/BASE_WA_PGT'

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
    --adapter --adapter_choose adapter da_head spatial_fusion_module global_fusion_module saf_module cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
    --load-from "outputs/BASE/latest.pth" --seed 134084244 \
