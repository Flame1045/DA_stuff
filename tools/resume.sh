
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
# WORKDIR="outputs/ONEB3_sim2city_unsupervised_woA"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --resume-from $WORKDIR/latest.pth --seed 134084244 \


# CONFIG="/media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/outputs/BASE_WA_Test42_Batch8/custom_sim2city_unsupervised_base_wA_woCTBV2.py"
# WORKDIR='outputs/BASE_WA_Test42_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --resume-from $WORKDIR/latest.pth --seed 134084244 \


######ADAPTERTest52####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="/media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/outputs/BASE_WA_Test52_Batch8/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
# WORKDIR='outputs/BASE_WA_Test52_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \
#     model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_res_adapter', 'norm','ffn', 'adapter_natten_V2a5x5', 'norm')" \
#     --resume-from $WORKDIR/latest.pth --seed 134084244 \
# #####TEST#####
# python3  tools/test.py "outputs/BASE_WA_Test52_Batch8/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py" \
#          "outputs/BASE_WA_Test52_Batch8/iter_11100.pth" \
#          --eval bbox
            ###############################################################################
         #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.705
            ###############################################################################
#####TEST#####

CONFIG="/media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/outputs/BASE_WA_Test60++3_Batch4/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
WORKDIR="outputs/BASE_WA_Test60++3_Batch4"
python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
    --adapter --adapter_choose SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
    --resume-from $WORKDIR/latest.pth --seed 134084244 \
