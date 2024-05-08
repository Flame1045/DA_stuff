######ADAPTERTest51####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTB_C2F.py"
# WORKDIR='outputs/BASE_WA_Test51_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F/latest.pth" --seed 134084244 --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \
#####TEST#####
# python3  tools/test.py "outputs/BASE_WA_Test51_Batch8/custom_sim2city_unsupervised_base_wA_woCTB_C2F.py" \
#          "outputs/BASE_WA_Test51_Batch8/iter_8500.pth" \
#          --eval bbox
            ###############################################################################
         #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.705
            ###############################################################################
#####TEST#####

######ADAPTERTest55####BASE_RES50###### Dcls 0to1 loss, Adapter in DEC 
# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_base_wA_woCTB_C2F.py"
# WORKDIR='outputs/BASE_WA_Test55_Batch8'

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH

# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
#     --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
#     --load-from "outputs/BASE_RES50_C2F__/iter_50592.pth" --seed 134084244 --cfg-options optimizer.lr="0.0002" optimizer.weight_decay="0.0000001" \

######ADAPTERTest58+2++####BASE_RES50######
CONFIG="/media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/outputs/BASE_WA_Test58+2_Batch4/custom_sim2city_unsupervised_base_wA_woCTBV2_C2F.py"
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

LR_VALUES=(0.0005 0.00007 0.00005)
MM_VALUES=(0.9 0.99 0.999)
for ((i=1; i<=3; i++)); do
   LR=${LR_VALUES[$i - 1]}  # Subtracting 1 because array index starts from 0
   for ((j=1; j<=3; j++)); do
      MM=${MM_VALUES[$j - 1]} 
      WORKDIR="outputs/BASE_WA_Test58+2++LR${LR}++MM${MM}_Batch2"
      python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --deterministic \
         --adapter --adapter_choose adapter scalar da_head cls_branches reg_branches label_embedding rpn_head roi_head bbox_head \
         --load-from "outputs/BASE_WA_Test58+2_Batch4/latest.pth" --seed 134084244 \
         --cfg-options optimizer.type="SGD" optimizer.momentum="$MM" optimizer.lr="$LR" \

   done
done

