CONFIG=$1
GPUS=$2
WEIGHTS=$3
WORKDIR=$4

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./mmselfsup/tools/analysis_tools/visualize_tsne.py $CONFIG \
    --checkpoint /media/ee4012/Disk3/Eric/Co-DETR/outputs/test/epoch12.pth --launcher pytorch ${@:4} --work-dir $WORKDIR
