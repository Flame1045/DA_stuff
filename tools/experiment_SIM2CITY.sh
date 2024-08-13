#############Ds. (SIM) → Dt.(Cityscapes)#############
#############Ds. (SIM) → Dt.(Cityscapes)#############
#############Ds. (SIM) → Dt.(Cityscapes)#############
##Deformable DETR: Deformable Transformers for End-to-End Object Detection##


#####SIM2CITY_baseline_and_pretrained_on_source

# CONFIG="experiment_saved/SIM2CITY_baseline_and_pretrained_on_source/custom_sim2city_base.py"
# WORKDIR='outputs/SIM2CITY_baseline_and_pretrained_on_source'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#       --seed 1764350285

# python3 tools/test.py experiment_saved/SIM2CITY_baseline_and_pretrained_on_source/custom_sim2city_base.py experiment_saved/SIM2CITY_baseline_and_pretrained_on_source/iter_45416.pth --eval bbox

#####SIM2CITY_oracle_and_trained_on_source_and_target

# CONFIG="experiment_saved/SIM2CITY_oracle_and_trained_on_source_and_target/custom_sim2city_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py"
# WORKDIR='outputs/SIM2CITY_oracle_and_trained_on_source_and_target'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#       --ORACLE \
#       --seed 1764350285

# python3 tools/test.py experiment_saved/SIM2CITY_oracle_and_trained_on_source_and_target/custom_sim2city_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py experiment_saved/SIM2CITY_oracle_and_trained_on_source_and_target/iter_43800.pth --eval bbox


#####SIM2CITY_with_Dcls_channel_mixing_spatail_attention
# CONFIG="experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/SIM2CITY_with_Dcls_channel_mixing_spatail_attention+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/pretrained/SIM2CITY.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
#          model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" 

# done

# python3 tools/test.py experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/iter_23600.pth --eval bbox

# python3 tools/test.py \
#       experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py \
#       experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/iter_23600.pth --eval bbox --show \
#       --show-score-thr 0.5 \
#       --show-dir experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/visaulization

######SIM2CITY_with_Dcls
# CONFIG="experiment_saved/SIM2CITY_with_Dcls/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/SIM2CITY_with_Dcls+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose slideatten adapter scalar SAP cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "experiment_saved/SIM2CITY_with_Dcls/pretrained/SIM2CITY.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde_nomlp, 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" 

# done
# python3 tools/test.py experiment_saved/SIM2CITY_with_Dcls/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py experiment_saved/SIM2CITY_with_Dcls/iter_5700.pth --eval bbox


######SIM2CITY_with_Dcls_channel_mixing
# CONFIG="experiment_saved/SIM2CITY_with_Dcls_channel_mixing/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/SIM2CITY_with_Dcls_channel_mixing+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose adapter scalar SAP cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "experiment_saved/SIM2CITY_with_Dcls_channel_mixing/pretrained/SIM2CITY.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde, 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nosilde', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" 

# done
# python3 tools/test.py experiment_saved/SIM2CITY_with_Dcls_channel_mixing/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py experiment_saved/SIM2CITY_with_Dcls_channel_mixing/iter_5700.pth --eval bbox


######SIM2CITY_with_Dcls_spatail_attention
# CONFIG="experiment_saved/SIM2CITY_with_Dcls_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# # Define the learning rates as an array
# LR_VALUES=(0.001)

# for ((i=1; i<=1; i++)); do
#       WORKDIR="outputs/SIM2CITY_with_Dcls_spatail_attention+${i}"
#       # Choose LR sequentially from the array using the loop index
#       LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
#       echo "Running iteration $i with WORKDIR=$WORKDIR"
#       python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#          --adapter --adapter_choose adapter scalar SAP cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#          --load-from "experiment_saved/SIM2CITY_with_Dcls_spatail_attention/pretrained/SIM2CITY.pth" --seed 134084244 \
#          --cfg-options optimizer.lr="$LR" \
#          model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nomlp, 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8_nomlp', 'norm', 'ffn', 'adapter', 'norm')" \
#          model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" 

# done
# python3 tools/test.py experiment_saved/SIM2CITY_with_Dcls_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py experiment_saved/SIM2CITY_with_Dcls_spatail_attention/iter_10500.pth --eval bbox

#####SIM2CITY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL
CONFIG="experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py"
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
# Define the learning rates as an array
LR_VALUES=(0.001)

for ((i=1; i<=1; i++)); do
      WORKDIR="outputs/SIM2CITY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL"
      # Choose LR sequentially from the array using the loop index
      LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
      echo "Running iteration $i with WORKDIR=$WORKDIR"
      python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
         --adapter --adapter_choose slideatten SAP adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
         --TINY_GT_LABEL \
         --load-from "experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/pretrained/SIM2CITY.pth" --seed 134084244 \
         --cfg-options optimizer.lr="$LR" optimizer.weight_decay="0.0000001" \
         model.query_head.transformer.decoder.transformerlayers.0.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.0.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.1.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.1.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.2.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.2.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.3.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.3.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.4.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.4.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.decoder.transformerlayers.5.operation_order="('self_attn', 'cross_attn_seq_adapterV25x5_slide8', 'norm', 'ffn', 'adapter', 'norm')" \
         model.query_head.transformer.encoder.transformerlayers.5.operation_order="('self_attn', 'norm', 'ffn', 'adapter', 'norm')" 

done
