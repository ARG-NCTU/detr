# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import cv2
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, AP_path):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    with open(AP_path, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        coco_evaluator.summarize()
        sys.stdout = original_stdout

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

################## Inference functions ##################

from torchvision import transforms
import numpy as np
import time

def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().strip().split('\n')
    return {i: label for i, label in enumerate(labels)}

@torch.no_grad()
def preprocess(frame, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame).unsqueeze(0).to(device)
    # print("Input tensor shape:", frame.shape)
    return frame

@torch.no_grad()
def perform_inference(frame, model, input_tensor, postprocessors, device):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    # print("Model outputs:", outputs)

    targets = [{'orig_size': torch.tensor([frame.shape[0], frame.shape[1]]), 
                'size': torch.tensor([input_tensor.shape[2], input_tensor.shape[3]])}]
    
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
    results = postprocessors['bbox'](outputs, orig_target_sizes)
    # print("Results:", results)
    
    if 'segm' in postprocessors.keys():
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        segm_results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # print("Segmentation Results:", segm_results)
        return results, segm_results
    return results, None

def draw_detections(frame, results, segm_results, classes_path, confidence_threshold):
    LABELS = load_labels(classes_path)
    mask_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Define colors for each class
    COLORS = {
        'Civilian_ship': (0, 0, 255),  # Red
        'Buoy': (57, 255, 20),         # Neon Green
        'Warship': (0, 0, 0),          # Black
        'Coast_guard': (255, 0, 0)     # Blue
    }

    for result in results:
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']

        for box, score, label in zip(boxes, scores, labels):
            if score > confidence_threshold:
                label_name = LABELS.get(label.item(), "Unknown")
                color = COLORS.get(label_name, (0, 255, 0))  # Default color if label not in COLORS

                # Draw rectangle and label text with specified color
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f'{label_name}: {score:.2f}', (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if segm_results is not None:
        for segm in segm_results:
            masks = segm['masks']
            scores = segm['scores']
            labels = segm['labels']
            for mask, score, label in zip(masks, scores, labels):
                if score > confidence_threshold:
                    label_name = LABELS.get(label.item(), "Unknown")
                    color = COLORS.get(label_name, (0, 0, 255))  # Default to red for masks if label not in COLORS

                    # Apply mask and contours
                    mask = mask.squeeze().cpu().numpy()
                    mask_image[mask > 0.1] = 255
                    contours, _ = cv2.findContours((mask > 0.1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, color, 2)

    return frame, mask_image


@torch.no_grad()
def padding_image(image, target_ratio):
    h, w = image.shape[:2]
    aspect_ratio_image = w / h
    pad_x, pad_y = 0, 0  # Initialize padding variables

    # Determine padding needed based on target ratio
    if aspect_ratio_image > target_ratio:
        new_h = int(w / target_ratio)
        pad_y = (new_h - h) // 2
        image = cv2.copyMakeBorder(image, pad_y, new_h - h - pad_y, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        new_w = int(h * target_ratio)
        pad_x = (new_w - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad_x, new_w - w - pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image, pad_x, pad_y


@torch.no_grad()
def depadding_image(image, target_ratio):
    h, w = image.shape[:2]
    aspect_ratio_image = w / h

    # 如果圖片的寬高比大於目標寬高比，則裁剪左右邊；否則裁剪上下邊
    if aspect_ratio_image > target_ratio:
        # Crop left and right
        new_w = int(h * target_ratio)
        start_w = (w - new_w) // 2
        image = image[:, start_w:start_w + new_w]
    elif aspect_ratio_image < target_ratio:
        # Crop top and bottom
        new_h = int(w / target_ratio)
        start_h = (h - new_h) // 2
        image = image[start_h:start_h + new_h, :]
    
    return image


@torch.no_grad()
def bboxes_proprocessing(bboxes, original_size, depadded_size, target_size, pad_x, pad_y):
    """
    Adjust bounding boxes based on resizing and depadding steps.
    - `original_size`: (height, width) of the original image
    - `depadded_size`: (height, width) of the image after depadding
    - `target_size`: (height, width) of the image after padding and resizing to target size
    - `pad_x`, `pad_y`: padding applied to width and height, respectively
    """
    # Ensure bboxes are float and reshaped correctly
    bboxes = bboxes.float().reshape(-1, 4)
    
    # Step 1: Adjust for the resizing to the target_size
    scale_x = depadded_size[1] / target_size[1]
    scale_y = depadded_size[0] / target_size[0]
    bboxes[:, [0, 2]] *= scale_x
    bboxes[:, [1, 3]] *= scale_y
    
    # Step 2: Reverse the depadding by removing padding offsets
    bboxes[:, [0, 2]] -= pad_x
    bboxes[:, [1, 3]] -= pad_y

    # Clip to ensure bboxes are within original image bounds
    bboxes[:, 0] = bboxes[:, 0].clamp(0, original_size[1])
    bboxes[:, 1] = bboxes[:, 1].clamp(0, original_size[0])
    bboxes[:, 2] = bboxes[:, 2].clamp(0, original_size[1])
    bboxes[:, 3] = bboxes[:, 3].clamp(0, original_size[0])

    return bboxes

@torch.no_grad()
def image_inference_test(frame, model, postprocessors, device, classes_path, confidence_threshold=0.5):
    original_size = (frame.shape[0], frame.shape[1])
    # print("Original frame shape:", frame.shape)
    target_ratio = 800 / 800
    frame, pad_x, pad_y = padding_image(frame, target_ratio)
    # print("Padded frame shape:", frame.shape)
    padded_size = (frame.shape[0], frame.shape[1])
    frame = cv2.resize(frame, (800, 800))

    # Simulate results
    results = [{'boxes': torch.tensor([[380, 380, 420, 420]]), 'scores': torch.tensor([0.99]), 'labels': torch.tensor([0])}]
    segm_results = None

    # Ensure results[0]['boxes'] is processed as 2D tensor
    results[0]['boxes'] = bboxes_proprocessing(results[0]['boxes'], original_size, padded_size, (800, 800), pad_x, pad_y)
    
    # Depadding and resizing
    frame = cv2.resize(frame, padded_size)
    target_ratio = original_size[1] / original_size[0]
    frame = depadding_image(frame, target_ratio)

    # Draw detections
    frame, mask_image = draw_detections(frame, results, segm_results, classes_path, confidence_threshold)

    # Dummy FPS calculation
    inference_time = 1  # Dummy time, replace with actual if needed
    current_fps = 1 / inference_time
    cv2.putText(frame, f'FPS: {current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return frame, mask_image

@torch.no_grad()
def image_inference(frame, model, postprocessors, device, classes_path, confidence_threshold=0.5):
    frame_out = frame.copy()
    original_size = (frame.shape[0], frame.shape[1])
    # print("Original frame shape:", frame.shape)
    target_ratio = 800 / 800
    frame, pad_x, pad_y = padding_image(frame, target_ratio)
    # print("Padded frame shape:", frame.shape)
    padded_size = (frame.shape[0], frame.shape[1])
    frame = cv2.resize(frame, (800, 800))
    
    input_tensor = preprocess(frame, device)
    start_time = time.time()
    results, segm_results = perform_inference(frame, model, input_tensor, postprocessors, device)
    end_time = time.time()
    
    # Ensure results[0]['boxes'] is processed as 2D tensor
    results[0]['boxes'] = bboxes_proprocessing(results[0]['boxes'], original_size, padded_size, (800, 800), pad_x, pad_y)

    # Ensure `boxes` is 2D for drawing detections
    if results[0]['boxes'].dim() == 1:
        results[0]['boxes'] = results[0]['boxes'].unsqueeze(0)
    
    # # Depadding and resizing
    # frame = cv2.resize(frame, padded_size)
    # target_ratio = original_size[1] / original_size[0]
    # frame = depadding_image(frame, target_ratio)
    
    # Draw detections on the depadded frame
    frame, mask_image = draw_detections(frame_out, results, segm_results, classes_path, confidence_threshold)
    
    # Calculate and display FPS
    inference_time = end_time - start_time
    current_fps = 1 / inference_time
    cv2.putText(frame, f'FPS: {current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return frame, mask_image



@torch.no_grad()
def video_inference(input_video_path, model, postprocessors, device, output_video_path, classes_path, output_mask_video_path=None, confidence_threshold=0.95):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if output_mask_video_path is not None:
        out_mask = cv2.VideoWriter(output_mask_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    count_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while cap.isOpened():
        print(f'Processing frame {count}/{count_max}', end="\r")
        ret, frame = cap.read()
        if not ret:
            break
        frame, mask_image = image_inference(frame, model, postprocessors, device, classes_path, confidence_threshold)
        # frame, mask_image = image_inference_test(frame, model, postprocessors, device, classes_path, confidence_threshold)
        out.write(frame)
        if output_mask_video_path is not None:
            out_mask.write(mask_image)
        
        if count % 100 == 0:
            cv2.imwrite(f'{output_video_path}_images/{count}.png', frame)
            if output_mask_video_path is not None:
                cv2.imwrite(f'{output_video_path}_images/{count}_mask.png', mask_image)
        count += 1
        
    cap.release()
    out.release()
    if output_mask_video_path is not None:
        out_mask.release()