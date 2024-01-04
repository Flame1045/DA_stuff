CONFIG=$1
WORKDIR=$2

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python3 $(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --adapter --adapter_choose adapter da_head scalar
