######ADAPTERTest52####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
# WORKDIR='outputs/BASE_WA_Test52_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F/latest.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \
#     model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_res_adapter', 'norm','ffn', 'adapter_natten_V2a5x5', 'norm')" \
#####TEST#####
# python3  tools/test.py "outputs/BASE_WA_Test52_Batch8/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py" \
#          "outputs/BASE_WA_Test52_Batch8/iter_11100.pth" \
#          --eval bbox
            ###############################################################################
         #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.705
            ###############################################################################
#####TEST#####


######ADAPTERTest56####BASE_RES50###### 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
# WORKDIR='outputs/BASE_WA_Test56_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \
#     model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_res_adapter', 'norm','ffn', 'adapter_natten_V2a5x5', 'norm')" \


######ADAPTERTest57####BASE_RES50###### 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
# WORKDIR='outputs/BASE_WA_Test57_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#     --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \
#     model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5', 'norm','ffn', 'adapter', 'norm')" \

######ADAPTERTest58+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.0005 0.001 0.00007 0.00005)

# for ((i=2; i<=5; i++)); do
#       WORKDIR="outputs/BASE_WA_Test58+${i}_Batch4"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 2]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          batch_size="2" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5', 'norm','ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \

# done

######ADAPTERTest59+iter####BASE_RES50######
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
# Define the learning rates as an array
LR_VALUES=(0.00002 0.0005 0.001 0.00007 0.00005)

for ((i=1; i<=1; i++)); do
      WORKDIR="outputs/BASE_WA_Test59++${i}_Batch4"
      # Choose LR sequentially from the array using the loop index
      LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
      echo "Running iteration $i with WORKDIR=$WORKDIR"
      python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
         --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
         --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
         --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
         model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')" \

done