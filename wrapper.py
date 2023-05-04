import os
import glob
import numpy as np
from tqdm import tqdm
from model import get_testing_model_resnet101
from segment import process, process_parts

def compute_part_features(images, scales=None, target_size=15, model_weights_path=None):
    if isinstance(images, str):
        images = sorted(glob.glob(os.path.join(images, '*')))
    if scales is None:
        scales = [1.0]
    if model_weights_path is None:
        model_weights_path = os.path.expanduser('~/remote_mounts/pomcloud0/DNN_weights/cdcl_model/model_simulated_RGB_mgpu_scaling_append.0071.h5')
    # load model
    model = get_testing_model_resnet101() 
    model.load_weights(model_weights_path)

    all_parts = []
    # generate image with body parts
    for image in tqdm(images):
        seg = process(image, model, scales=scales)
        parts = process_parts(seg, target_size=target_size)
        all_parts.append(parts)
    return np.array(all_parts)