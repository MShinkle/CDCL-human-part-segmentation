import numpy as np
import cv2
import os
from tqdm import tqdm
from .model import get_testing_model_resnet101
from .segment import process, process_parts

class CDCL_process():
    def __init__(self):
        self.model = None

    def download_weights(self, weights_dir='weights', verbose=True):
        if verbose:
            print('Downloading weights')
        os.system('wget -O {}/cdcl_model.zip https://www.dropbox.com/s/sknafz1ep9vds1r/cdcl_model.zip?dl=1'.format(weights_dir))
        os.system('unzip {}/cdcl_model.zip -d {}'.format(weights_dir, weights_dir))
        os.system('rm {}/cdcl_model.zip'.format(weights_dir))

    def load_model(self, model_weights_path='weights/model_simulated_RGB_mgpu_scaling_append.0071.h5', verbose=True):
        if verbose:
            print('Loading model')
        if not os.path.exists(model_weights_path):
            self.download_weights(weights_dir=os.path.dirname(model_weights_path), verbose=verbose)
        weights = os.path.expanduser(model_weights_path)
        self.model = get_testing_model_resnet101() 
        self.model.load_weights(weights)

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