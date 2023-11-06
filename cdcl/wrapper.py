import numpy as np
import cv2
from tqdm import tqdm
from .model import get_testing_model_resnet101
from .segment import process, process_parts

class CDCL_process():
    def __init__(self):
        self.model = None

    def load_model(self, model_weights_path='d:/DNN_weights/cdcl_model/model_simulated_RGB_mgpu_scaling_append.0071.h5', verbose=True):
        if verbose:
            print('Loading model')
        self.model = get_testing_model_resnet101() 
        self.model.load_weights(model_weights_path)

    def extract_parts(self, images, scales, target_size, thresholds='default', verbose=True, input_rgb=True):
        if self.model is None:
            self.load_model(verbose=verbose)
        if images.ndim==3:
            images = images[np.newaxis]
        if input_rgb:
            images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images])
        if np.issubdtype(images.dtype, np.floating) and images.max()<=1:
            images *= 255
            images = images.astype(np.uint8)
        if verbose:
            images = tqdm(images)
        all_parts = []
        for image in images:
            segmentation = process(image, self.model, scales=scales)
            parts = process_parts(segmentation, target_size=target_size, thresholds=thresholds)
            all_parts.append(parts)
        return np.array(all_parts)