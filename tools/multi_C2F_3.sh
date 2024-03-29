######ADAPTERTest52####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
WORKDIR='outputs/BASE_WA_Test52_Batch8'

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
    --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
    --load-from "outputs/BASE_RES50_C2F/latest.pth" --seed 134084244 \
    --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \
    model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_res_adapter', 'norm','ffn', 'adapter_natten_V2a5x5', 'norm')" \