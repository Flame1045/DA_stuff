#####20240328#####source only BASE_RES50#####

# CONFIG="projects/configs/co_dino/custom_sim2city_base_C2F.py"
# WORKDIR='outputs/BASE_RES50_C2F'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic

#####ADAPTERTest50####BASE_RES50###### Dcls 0to1 loss 
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woAnCTB_C2F.py"
WORKDIR='outputs/BASE_WA_Test50_Batch8'

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
    --adapter --adapter_choose da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
    --load-from "outputs/BASE_RES50_C2F/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \

