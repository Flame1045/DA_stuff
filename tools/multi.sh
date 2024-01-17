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

CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_woA.py"
WORKDIR3="outputs/custom_sim2city_unsupervised_lamda_scheduler01_da_head_only_scalar_new256D_after_transformer"

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR3 --deterministic

##################

# CONFIG="projects/configs/co_dino/custom_sim2city_unsupervised_wA.py"
# WORKDIR4="outputs/custom_sim2city_unsupervised_lamda_scheduler01_da_head_adapter_only_scalar_new256D"

# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# echo $PYTHONPATH
# python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR4 --load-from "outputs/custom_sim2city_unsupervised_lamda_scheduler01_da_head_all_scalar_new256D/iter_28000.pth" \
# --deterministic --adapter --adapter_choose adapter da_head scalar