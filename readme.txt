git clone git@github.com:ARG-NCTU/detr.git

cd detr

source Docker/docker_run.sh

## Training

python -m torch.distributed.launch --batch_size 4 --nproc_per_node=1 --use_env main.py --coco_path COCO_2017 --output_dir output --lr_drop 100 --epochs 150

## Evaluating

python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path COCO_2017