
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
# WORKDIR="outputs/ONEB3_sim2city_unsupervised_woA"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --resume-from $WORKDIR/latest.pth --seed 134084244 \


CONFIG="/media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/outputs/BASE_WA_Test42_Batch8/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
WORKDIR='outputs/BASE_WA_Test42_Batch8'

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
    --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
    --resume-from $WORKDIR/latest.pth --seed 134084244 \