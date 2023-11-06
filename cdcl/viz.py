import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

def show_parts_overlay(seg, bg_img=None, part_colors=('magenta', 'yellow', 'blue', 'cyan', 'red', 'lightgreen')):
    seg = seg.squeeze()
    if part_colors is None:
        part_colors = ['white']*len(seg)
    colored_parts = []
    for part, color in zip(seg, part_colors):
        if isinstance(color, str):
            color_values = np.array(colors.to_rgba(color))[np.newaxis,:3]
        else:
            color_values = color
        colored_part = part[...,np.newaxis]*color_values
        colored_parts.append(colored_part)
    colored_parts = np.sum(colored_parts, axis=0)
    colored_parts = np.clip(colored_parts, 0, 1)
    if isinstance(bg_img, str):
        bg_img = plt.imread(bg_img)
    if isinstance(bg_img, np.ndarray):
        plt.imshow(bg_img, extent=[0,1,0,1])
    plt.imshow(colored_parts, extent=[0,1,0,1], alpha=.5 if isinstance(bg_img, np.ndarray) else 1)
    plt.xticks([])
    plt.yticks([])