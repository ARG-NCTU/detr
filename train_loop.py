import os 

# Function to find the latest checkpoint
def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint0")]
    if checkpoints:
        # Sort checkpoints based on the epoch number and return the latest one
        checkpoints = sorted(checkpoints, key=lambda x: x.split("checkpoint0")[-1])
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming from the latest checkpoint: {latest_checkpoint}")
        return os.path.join(output_dir, latest_checkpoint)

output_dir = "output/boat-1107-1000-epochs"
# latest_checkpoint = get_latest_checkpoint(output_dir)
# os.system(f"python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py -- --resume output/boat-1101-100-epochs/checkpoint0028.pth --coco_path Kaohsiung_Port_dataset --output_dir output/boat-1104-600-epochs --lr_drop 300 --epochs 600 --lr_reset True")

for i in range(100):
    latest_checkpoint = get_latest_checkpoint(output_dir)
    os.system(f"python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py -- --resume {latest_checkpoint} --coco_path Kaohsiung_Port_dataset --output_dir output/boat-1107-1000-epochs --lr 1e-5 --lr_drop 600 --epochs 1000 --lr_reset True")
