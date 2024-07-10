#############Ds. (SIM) → Dt.(Cityscapes)#############
#############Ds. (SIM) → Dt.(Cityscapes)#############
#############Ds. (SIM) → Dt.(Cityscapes)#############
##Deformable DETR: Deformable Transformers for End-to-End Object Detection##


#####20240311#####source only BASE_RES50##### 

# CONFIG="projects/configs/co_dino/custom_sim2city_base.py"
# WORKDIR='outputs/BASE_RES50'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#       --seed 1764350285

######ADAPTERTest77ABA_wo_adapter+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test77ABA_wo_adapter+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR"

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
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          lr_config.step="[500]" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"


# done

######ADAPTERTest77+iter####BASE_RES50###### SAP  Numbers of 20304725 _is_trained BEST
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
# Define the learning rates as an array
LR_VALUES=(0.001)

for ((i=1; i<=1; i++)); do
      WORKDIR="outputs/BASE_WA_Test77+${i}"
      # Choose LR sequentially from the array using the loop index
      LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
      echo "Running iteration $i with WORKDIR=$WORKDIR"
      python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
         --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
         --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
         --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" model.aroiweight="1.0" model.gamma="0.5" \
         model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \

done

######ADAPTERTest86####BASE_RES50###### # SAP  Numbers of 20304725 _is_trained
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test86+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose adapter scalar SAP cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8, 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \


# done

######ADAPTERTest86####BASE_RES50###### # SAP  Numbers of 20304725 _is_trained
CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
# Define the learning rates as an array
LR_VALUES=(0.001)

for ((i=1; i<=1; i++)); do
      WORKDIR="outputs/BASE_WA_Test86+${i}"
      # Choose LR sequentially from the array using the loop index
      LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
      echo "Running iteration $i with WORKDIR=$WORKDIR"
      python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
         --adapter --adapter_choose adapter scalar SAP cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
         --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
         --cfg-options optimizer.lr="$LR" \
         model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \


done


# ######ADAPTERTest86####BASE_RES50###### # DAAN _nosilde
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test86+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          lr_config.step="[500]" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"


# done

######ADAPTERTest87####BASE_RES50###### # DAAN _nomlp
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test87+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          lr_config.step="[500]" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"


# done

######ADAPTERTest88####BASE_RES50###### # DAAN _nosilde_nomlp
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test88+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50/iter_45416.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          lr_config.step="[500]" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8_nosilde_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"


# done


#############Ds. (Cityscapes) → Dt.(Foggy)#############
#############Ds. (Cityscapes) → Dt.(Foggy)#############
#############Ds. (Cityscapes) → Dt.(Foggy)#############

#####20240328#####source only BASE_RES50##### 

# CONFIG="projects/configs/co_dino/custom_sim2city_base_C2F.py"
# WORKDIR='outputs/BASE_RES50_C2F__'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --cfg-options epochs=40 --seed 129094272 


######ADAPTERTest82+iter####BASE_RES50######
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/BASE_WA_Test82+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 \
#          --cfg-options optimizer.type="SGD" optimizer.lr="$LR" optimizer.weight_decay="0.0000001" optimizer.momentum="0.9" \
#          lr_config.step="[5000, 11000]" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')"

# done