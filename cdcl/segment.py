import cv2
import numpy as np
import skimage
import tensorflow as tf

human_part = [0,1,2,4,3,6,5,8,7,10,9,12,11,14,13]
human_ori_part = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

default_thresholds = {
    'background': .99,
    'head': .99,
    'torso': .99,
    'rightupperarm': .99,
    'leftupperarm': .99,
    'rightforearm': .99,
    'leftforearm': .99,
    'lefthand': .99,
    'righthand': .99,
    'leftthigh': .99,
    'rightthigh': .99,
    'leftshank': .99,
    'rightshank': .99,
    'rightfoot': .99 ,
    'leftfoot': .99,
}

def recover_flipping_output(image, part_ori_size):
    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((image.shape[0], image.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return part_flip_size

def process(input_image, model, scales):
    stride=8
    boxsize=368
    input_image = (input_image / 256.0) - 0.5
    scales = [x * boxsize / input_image.shape[0] for x in scales]
    seg_maps = []
    for scale in scales:
        for flip in [False, True]:
            image = input_image.copy()
            if flip:
                image = cv2.flip(image, 1)
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            pad_vertical   = (image.shape[0] - stride) % stride
            pad_horizontal = (image.shape[1] - stride) % stride
            image = np.pad(image, ((0, pad_vertical), (0, pad_horizontal), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
            output_blobs = model.predict_on_batch(image[np.newaxis])
            seg = np.squeeze(output_blobs[2])
            seg = cv2.resize(seg, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            seg = seg[:image.shape[0] - pad_vertical, :image.shape[1] - pad_horizontal, :]
            seg = cv2.resize(seg, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)
            if flip:
                seg = recover_flipping_output(input_image, seg)
            seg_maps.append(seg)
    seg_avg = np.max(seg_maps, axis=0)
    return seg_avg

part_idxs = np.array([[1,1,1,1], [2,2,2,2], [3,4,5,6,], [7,7,8,8], [9,10,11,12], [13,13,14,14]])

def process_parts(seg, target_size=15, thresholds=None, pooling_fn=np.nanmax):
    if thresholds is not None:
        if isinstance(thresholds, str):
            thresholds = default_thresholds
        if isinstance(thresholds, dict):
            thresholds = [thresholds[part] for part in default_thresholds.keys()]
        if isinstance(thresholds, (list, tuple)):
            thresholds = np.array(thresholds)
        seg = seg > thresholds
    if isinstance(target_size, (int)):
        target_size = (target_size, target_size)
    parts = seg[:,:,part_idxs].max(-1)
    parts = np.moveaxis(parts,2,0)
    if target_size is not None:
        block_size_vertical = int(np.ceil(parts.shape[1]/target_size[0]))
        block_size_horizontal = int(np.ceil(parts.shape[2]/target_size[1]))
        resized = np.array([skimage.measure.block_reduce(part, (block_size_vertical, block_size_horizontal), pooling_fn) for part in parts])
        parts = resized.astype(np.float32)
    return parts
