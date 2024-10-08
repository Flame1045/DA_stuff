#####20240311#####source only BASE_RES50##### Ds. (SIM) → Dt.(Cityscapes)

# CONFIG="projects/configs/co_dino/custom_sim2city_base.py"
# WORKDIR='outputs/BASE_RES50'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#       --seed 1764350285


#####20240328#####source only BASE_RES50##### Ds. (Cityscapes) → Dt.(Foggy)

# CONFIG="projects/configs/co_dino/custom_sim2city_base_C2F.py"
# WORKDIR='outputs/BASE_RES50_C2F__'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --cfg-options epochs=40 --seed 129094272 


#####ADAPTERTest50####BASE_RES50###### Dcls 0to1 loss 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woAnCTB_C2F.py"
# WORKDIR='outputs/BASE_WA_Test50_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \

#####TEST#####
# python3  tools/test.py "outputs/BASE_WA_Test50_Batch8/custom_sim2city_unsupervised_base_woAnCTB_C2F.py" \
#          "outputs/BASE_WA_Test50_Batch8/iter_10400.pth" \
#          --eval bbox
            ###############################################################################
         #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.705
            ###############################################################################
#####TEST#####

#####ADAPTERTest53####BASE_RES53###### Dcls 0to1 loss 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woAnCTB_C2F.py"
# WORKDIR='outputs/BASE_WA_Test53_Batch4'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     # --adapter --adapter_choose query_head neck da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.000002" optimizer.weight_decay="0.0001" \

#####ADAPTERTest54####BASE_RES54###### Dcls 0to1 loss 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woAnCTB_C2F.py"
# WORKDIR='outputs/BASE_WA_Test54_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \

#####ADAPTERTest54####BASE_RES50###### Dcls 0to1 loss 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_woAnCTB_C2F.py"
# WORKDIR='outputs/BASE_WA_Test54_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \

######ADAPTERTest57+iter####BASE_RES50###### 
#!/bin/bash
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# for ((i=1; i<=10; i++)); do
#     WORKDIR="outputs/BASE_WA_Test57+${i}_Batch8"
#     echo "Running iteration $i with WORKDIR=$WORKDIR"
#     python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#         --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#         --load-from "outputs/BASE_WA_Test57_Batch8/iter_9500.pth" --seed 134084244 \
#         --cfg-options optimizer.type="SGD" optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" optimizer.momentum="0.9"\
#         model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5', 'norm','ffn', 'adapter', 'norm')"
# done

######ADAPTERTest58+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# for ((i=1; i<=1; i++)); do
#     WORKDIR="outputs/BASE_WA_Test58+${i}_Batch2"
#     echo "Running iteration $i with WORKDIR=$WORKDIR"
#     python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#         --adapter --adapter_choose adapter scalar da_head rpn_head roi_head \
#         --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#         --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \
#         batch_size="2" \
#         model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5', 'norm','ffn', 'adapter', 'norm')" \
#         model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \


# done

######ADAPTERTest60+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.00002 0.0005 0.001 0.00007 0.00005)

# for ((i=1; i<=5; i++)); do
#       WORKDIR="outputs/BASE_WA_Test60++${i}_Batch4"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5', 'norm', 'ffn', 'adapter', 'norm')" \

# done

######ADAPTERTest60+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.00002 0.0005 0.001 0.00007 0.00005)

# for ((i=3; i<=3; i++)); do
#       WORKDIR="outputs/BASE_WA_Test60+++${i}_Batch4"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose SAP adapter scalar cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done

######ADAPTERTest61+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.00002 0.0005 0.001 0.00007 0.00005)

# for ((i=3; i<=3; i++)); do
#       WORKDIR="outputs/Test61+"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done

######ADAPTERTest63+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.00002 0.0005 0.001 0.00007 0.00005)

# for ((i=3; i<=3; i++)); do
#       WORKDIR="outputs/Test63"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('slide_attn', 'cross_attn_seq_adapterV25x5', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('slide_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done

######ADAPTERTest64+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.00002 0.0005 0.001 0.00007 0.00005)

# for ((i=3; i<=3; i++)); do
#       WORKDIR="outputs/Test64"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done

######ADAPTERTest65+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.00002 0.0005 0.00007 0.00005)

# for ((i=1; i<=4; i++)); do
#       WORKDIR="outputs/BASE_WA_Test65+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done


######ADAPTERTest66+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test66+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done

######ADAPTERTest67+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test67+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done

######ADAPTERTest68+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test68+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="0.25" model.gamma="1.0" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done

##TEST##
# python3 tools/test.py outputs/BASE_WA_Test68+1/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py outputs/BASE_WA_Test68+1/iter_64800.pth --eval bbox

######ADAPTERTest69+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test69+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'seq_adapter', 'norm', 'ffn', 'res_adapter', 'norm')"

# done


######ADAPTERTest70+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test70+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest71+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test71+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest72+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/test/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4_da_head.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.00015)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test72+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest73+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/test/custom_sim2city_unsupervised_base_wA_woCTBV2_B4_da_head.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test73+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest74+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test74+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest77+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test77+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest75+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test75+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide16', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest78+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test78+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide16', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest79+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test78+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide16', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest80+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test80+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_WA_Test70+1/iter_51000.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest81+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test81+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_WA_Test70+1/iter_51000.pth" --seed 134084223 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

#####ADAPTERTestABA####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test81+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_WA_Test70+1/iter_51000.pth" --seed 134084223 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
#          model.query_head.transformer.decoder.transformerlayers.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done

######ADAPTERTest85####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test85+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --resume-from "outputs/BASE_WA_Test85+1/iter_500.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          lr_config.step="[500]" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"


# done

######ADAPTERTest84+iter####BASE_RES50######
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
# Define the learning rates as an array
LR_VALUES=(0.001)

for ((i=1; i<=1; i++)); do
      WORKDIR="outputs/BASE_WA_Test84+${i}"
      # Choose LR sequentially from the array using the loop index
      LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
      echo "Running iteration $i with WORKDIR=$WORKDIR"
      python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
         --adapter --adapter_choose slideatten adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
         --load-from "outputs/BASE_WA_Test81+1/iter_49900.pth" --seed 134084244 \
         --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
         model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

done