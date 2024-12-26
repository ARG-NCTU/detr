**DE⫶TR**: End-to-End Object Detection with Transformers
========

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)

PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
We replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

![DETR](.github/DETR.png)

**What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

**About the code**. We believe that object detection should not be more difficult than classification,
and should not require complex libraries for training and inference.
DETR is very simple to implement and experiment with, and we provide a
[standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)
showing how to do inference with DETR in only a few lines of PyTorch code.
Training code follows this idea - it is not a library,
but simply a [main.py](main.py) importing model and criterion
definitions with standard training loops.

Additionnally, we provide a Detectron2 wrapper in the d2/ folder. See the readme there for more information.

For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

See our [blog post](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers/) to learn more about end to end object detection with transformers.
# Model Zoo
We provide baseline DETR and DETR-DC5 models, and plan to include more in future.
AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images,
with torchscript transformer.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>

COCO val5k evaluation results can be found in this [gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).

The models are also available via torch hub,
to load DETR R50 with pretrained weights simply do:
```python
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
```


COCO panoptic val5k models:
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>box AP</th>
      <th>segm AP</th>
      <th>PQ</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>38.8</td>
      <td>31.1</td>
      <td>43.4</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth">download</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>40.2</td>
      <td>31.9</td>
      <td>44.6</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth">download</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>40.1</td>
      <td>33</td>
      <td>45.1</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth">download</a></td>
      <td>237Mb</td>
    </tr>
  </tbody>
</table>

Checkout our [panoptic colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb)
to see how to use and visualize DETR's panoptic segmentation prediction.

# Notebooks

We provide a few notebooks in colab to help you get a grasp on DETR:
* [DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb): Shows how to load a model from hub, generate predictions, then visualize the attention of the model (similar to the figures of the paper)
* [Standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb): In this notebook, we demonstrate how to implement a simplified version of DETR from the grounds up in 50 lines of Python, then visualize the predictions. It is a good starting point if you want to gain better understanding the architecture and poke around before diving in the codebase.
* [Panoptic Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb): Demonstrates how to use DETR for panoptic segmentation and plot the predictions.


# Usage - Object detection
There are no extra compiled components in DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via Docker (recommended) or conda.

## 1. Clone the repository
Clone the repository locally with ssh:
```bash
git clone git@github.com:ARG-NCTU/detr.git
```
If you have no ssh key, you can also clone with https:
```bash
git clone https://github.com/ARG-NCTU/detr.git
```

## 2. With Docker (recommended)
Install and run the docker image on desktop or laptop computer.
```bash
source Docker/gpu/run.sh
```
Or, install and run the docker image on jetson orin.
```bash
source Docker/jetson-orin/run.sh
```

## 2. With Conda (not recommended)
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## 3. Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```bash
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

Download and extract **virtual boat & real WAM-V** 42.8k train and 5.4k val images with annotations (60.74G) for finetuning.
```bash
wget ftp://140.113.148.83/arg-projectfile-download/DETR/dataset/Boat_dataset.zip
unzip Boat_dataset.zip
```
We expect the directory structure to be the following:
```bash
Boat_dataset/
  annotations/  # annotation json files
  train2024/    # train images
  val2024/      # val images
```

Download and extract **real boat** 601 train and 150 images with annotations (187.8MB) for finetuning.
```bash
wget ftp://140.113.148.83/arg-projectfile-download/DETR/dataset/Kaohsiung_Port_dataset.zip
unzip Kaohsiung_Port_dataset.zip
```
We expect the directory structure to be the following:
```bash
Kaohsiung_Port_dataset/
  annotations/  # annotation json files
  train2024/    # train images
  val2024/      # val images
```

Download and extract **fish jump** 1.7k train and 434 val images with annotations (318.1MB) for finetuning.
```bash
wget ftp://140.113.148.83/arg-projectfile-download/DETR/dataset/fish_jump_dataset_2024.zip
unzip fish_jump_dataset_2024.zip
```
We expect the directory structure to be the following:
```bash
fish_jump_dataset_2024/
  annotations/  # annotation json files
  train2024/    # train images
  val2024/      # val images
```

Download and extract **water splash** 874 train and 223 val images with annotations (491.6MB) for finetuning.
```bash
wget ftp://140.113.148.83/arg-projectfile-download/DETR/dataset/water_splash_dataset_2024.zip
unzip water_splash_dataset_2024.zip
```
We expect the directory structure to be the following:
```bash
water_splash_dataset_2024/
  annotations/  # annotation json files
  train2024/    # train images
  val2024/      # val images
```

## 4. Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```bash
python main.py --coco_path /path/to/coco --output_dir output/example
```

Finetuning pretrained model with **virtual boat & real WAM-V** dataset for example (more arguments setting is in main.py):
```bash
python main.py --resume pretrained_models/detr-r50-pretrained.pth --coco_path Kaohsiung_Port_dataset --output_dir output/boat-1115-600-epochs --lr_drop 300 --epochs 600
```

Finetuning pretrained model with **real boat** dataset for example (more arguments setting is in main.py):
```bash
python main.py --resume pretrained_models/detr-r50-pretrained.pth --coco_path Kaohsiung_Port_dataset --output_dir output/boat-1115-600-epochs --lr_drop 300 --epochs 600
```

Second finetuning pretrained model with **real boat** dataset for example (more arguments setting is in main.py):
```bash
python main.py --resume output/boat-1107-1000-epochs/checkpoint0642.pth --coco_path Kaohsiung_Port_dataset --output_dir output/boat-1107-1000-epochs --lr_drop 600 --epochs 1000 --lr_reset True
```

Finetuning pretrained model with **fish jump** dataset for example:
```bash
python main.py --resume pretrained_models/detr-r50-pretrained.pth --coco_path fish_jump_dataset_2024 --output_dir output/0612 --lr_drop 5 --epochs 20
```

Finetuning pretrained model with **water splash** dataset for example:
```bash
python main.py --resume pretrained_models/detr-r50-pretrained.pth --coco_path water_splash_dataset_2024 --output_dir output/bbox-0619 --lr_drop 200 --epochs 300
```

A single epoch takes 28 minutes, so 300 epoch training
takes around 6 days on a single machine with 8 V100 cards.
To ease reproduction of our results we provide
[results and training logs](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)
for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50.

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


## 5. Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```bash
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco --AP_path output/example/AP_summerize.txt
```

Evaluate finetuned model with **virtual boat & real WAM-V** dataset for example:
```bash 
python main.py --batch_size 2 --no_aux_loss --eval --resume output/0328/checkpoint.pth --coco_path Boat_dataset --AP_path output/0328/AP_summerize.txt
```

Evaluate finetuned model with **real boat** dataset for example:
```bash
python main.py --batch_size 2 --no_aux_loss --eval --resume output/boat-1107-1000-epochs/checkpoint0642.pth --coco_path Kaohsiung_Port_dataset --AP_path output/boat-1107-1000-epochs/AP_summerize.txt
```

Evaluate finetuned model with **fish jump** dataset for example:
```bash
python main.py --batch_size 2 --no_aux_loss --eval --resume output/0612/checkpoint.pth --coco_path fish_jump_dataset_2024 --AP_path output/0612/AP_summerize_real.txt
```

Evaluate finetuned model with **water splash** dataset for example:
```bash
python main.py --batch_size 2 --no_aux_loss --eval --resume output/bbox-0619/checkpoint.pth --coco_path water_splash_dataset_2024 --AP_path output/bbox-0619/AP_summerize_final.txt
```
We provide results for all DETR detection models in this
[gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).
Note that numbers vary depending on batch size (number of images) per GPU.
Non-DC5 models were trained with batch size 2, and DC5 with 1,
so DC5 models show a significant drop in AP if evaluated with more
than 1 image per GPU.

## 6. Inference
Download and extract all trained models weight
```bash
wget ftp://140.113.148.83/arg-projectfile-download/DETR/output.zip
unzip output.zip
```
Download and extract all source video 
```bash
wget ftp://140.113.148.83/arg-projectfile-download/DETR/source_video.zip
unzip source_video.zip
```

Inference finetuned model with **virtual boat & real WAM-V** image for example:
```bash
python main.py --inference_image --resume output/0328/checkpoint_0328.pth --input_image_path source_image/WamV.jpg --output_image_path output_image/WamV_out.jpg --classes_path Boat_dataset/classes.txt
```

Inference finetuned model with **virtual boat & real WAM-V** video for example:
```bash
python main.py --inference_video --resume output/0328/checkpoint_0328.pth --input_video_path source_video/WAM_V_1.mp4 --output_video_path output_video/WAM_V_1_out.mp4 --classes_path Boat_dataset/classes.txt
```

Inference finetuned model with **real boat** video for example:
```bash
python main.py --inference_video --resume output/boat-1104-600-epochs/checkpoint0672.pth --input_video_path source_video/kaohsiung_port1.mp4 --output_video_path output_video/kaohsiung_port1_out.mp4 --classes_path Kaohsiung_Port_dataset/annotations/classes.txt --confidence_thershold 0.8
```

Inference finetuned model with video for **fish jump** example:
```bash
python main.py --inference_video --resume output/0612/checkpoint.pth --input_video_path source_video/flow_1.mp4 --output_video_path output_video/flow_1_out.mp4 --classes_path fish_jump_dataset_2024/classes.txt
```

Inference finetuned bbox + segm model with video for **water splash** example:
```bash
python main.py --masks --inference_video --resume output/seg-0619/checkpoint.pth --frozen_weights output/seg-0619/checkpoint.pth --input_video_path source_video/aeratorcompare10M_flow_img.mp4 --output_video_path output_video/aeratorcompare10M_flow_img_out.mp4 --output_mask_video_path output_video/aeratorcompare10M_flow_img_segm_out.mp4 --classes_path water_splash_dataset_2024/classes.txt --confidence_thershold 0.95
```


## Multinode training
Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):
```
pip install submitit
```
Train baseline DETR-6-6 model on 4 nodes for 300 epochs:
```
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```

# Usage - Segmentation

We show that it is relatively straightforward to extend DETR to predict segmentation masks. We mainly demonstrate strong panoptic segmentation results.

## Data preparation

For panoptic segmentation, you need the panoptic annotations additionally to the coco dataset (see above for the coco dataset). You need to download and extract the [annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip).
We expect the directory structure to be the following:
```
path/to/coco_panoptic/
  annotations/  # annotation json files
  panoptic_train2017/    # train panoptic annotations
  panoptic_val2017/      # val panoptic annotations
```

## Training

We recommend training segmentation in two stages: first train DETR to detect all the boxes, and then train the segmentation head.
For panoptic segmentation, DETR must learn to detect boxes for both stuff and things classes. You can train it on a single node with 8 gpus for 300 epochs with:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco  --coco_panoptic_path /path/to/coco_panoptic --dataset_file coco_panoptic --output_dir /output/path/box_model
```
For instance segmentation, you can simply train a normal box model (or used a pre-trained one we provide).

Once you have a box model checkpoint, you need to freeze it, and train the segmentation head in isolation.
For panoptic segmentation you can train on a single node with 8 gpus for 25 epochs:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --masks --epochs 25 --lr_drop 15 --coco_path /path/to/coco  --coco_panoptic_path /path/to/coco_panoptic  --dataset_file coco_panoptic --frozen_weights /output/path/box_model/checkpoint.pth --output_dir /output/path/segm_model
```
For instance segmentation only, simply remove the `dataset_file` and `coco_panoptic_path` arguments from the above command line.

Finetuning pretrained model with **virtual boat & real WAM-V** dataset for example (more arguments setting is in main.py):
```bash 
python main.py --masks --epochs 10 --coco_path Boat_dataset --frozen_weights output/0328/checkpoint.pth --output_dir output/0613
```
Finetuning pretrained model with **fish jump** dataset for example:
```bash
python main.py --masks --epochs 10 --coco_path fish_jump_dataset_2024 --frozen_weights output/0612/checkpoint.pth --output_dir output/0613
```
Finetuning pretrained model with **water splash** dataset for example:
```bash
python main.py --masks --epochs 55 --coco_path water_splash_dataset_2024 --frozen_weights output/bbox-0619/checkpoint.pth --output_dir output/seg-0619
```

# License
DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
