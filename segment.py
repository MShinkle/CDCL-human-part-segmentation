import cv2
import numpy as np
import skimage

right_part_idx = [2, 3, 4,  8,  9, 10, 14, 16]
left_part_idx =  [5, 6, 7, 11, 12, 13, 15, 17]
human_part = [0,1,2,4,3,6,5,8,7,10,9,12,11,14,13]
human_ori_part = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
seg_num = 15 # current model supports 15 parts only

def recover_flipping_output(oriImg, part_ori_size):

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return part_flip_size

default_thresholds = {
    'background': 0.6,
    'head': 0.5,
    'torso': 0.8,
    'rightfoot': 0.55 ,
    'leftfoot': 0.55,
    'leftthigh': 0.55,
    'rightthigh': 0.55,
    'leftshank': 0.55,
    'rightshank': 0.55,
    'rightupperarm': 0.55,
    'leftupperarm': 0.55,
    'rightforearm': 0.55,
    'leftforearm': 0.55,
    'lefthand': 0.55,
    'righthand': 0.55,
}

def process(input_image, model, scales=None):
    input_scale = 1.0
    stride=8
    if scales is None:
        scales = [1.0]

    boxsize = 368

    if isinstance(input_image, str):
        oriImg = cv2.imread(input_image)
    else:
        oriImg = input_image
    flipImg = cv2.flip(oriImg, 1)
    oriImg = (oriImg / 256.0) - 0.5
    flipImg = (flipImg / 256.0) - 0.5
    multiplier = [x * boxsize / oriImg.shape[0] for x in scales]

    seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m]*input_scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [ 0,
                0, 
                (imageToTest.shape[0] - stride) % stride,
                (imageToTest.shape[1] - stride) % stride
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

        input_img = imageToTest_padded[np.newaxis, ...]

        output_blobs = model.predict(input_img)
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=stride, fy=stride,
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        if m==0:
            segmap_scale1 = seg
        elif m==1:
            segmap_scale2 = seg         
        elif m==2:
            segmap_scale3 = seg
        elif m==3:
            segmap_scale4 = seg


    # flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = [ 0,
                0, 
                (imageToTest.shape[0] - stride) % stride,
                (imageToTest.shape[1] - stride) % stride
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        input_img = imageToTest_padded[np.newaxis, ...]
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=stride, fy=stride,
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        seg_recover = recover_flipping_output(oriImg, seg)

        if m==0:
            segmap_scale5 = seg_recover
        elif m==1:
            segmap_scale6 = seg_recover         
        elif m==2:
            segmap_scale7 = seg_recover
        elif m==3:
            segmap_scale8 = seg_recover

    segmap_a = np.maximum(segmap_scale1,segmap_scale2)
    segmap_b = np.maximum(segmap_scale4,segmap_scale3)
    segmap_c = np.maximum(segmap_scale5,segmap_scale6)
    segmap_d = np.maximum(segmap_scale7,segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)
    return seg_avg

part_idxs = np.array([[1,1,1,1], [2,2,2,2], [3,4,5,6,], [7,7,8,8], [9,10,11,12], [13,13,14,14]])

def process_parts(seg, target_size=15):
    six_parts = seg[:,:,part_idxs].max(-1)
    block_size = int(np.ceil(six_parts.shape[1]/target_size))
    resized = np.array([skimage.measure.block_reduce(part, (block_size, block_size), np.nanmax) for part in np.moveaxis(six_parts,2,0)])
    parts = resized.astype(np.float32)
    return parts
