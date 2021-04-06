import os
import sys

import cv2
import numpy as np


sys.path.append("..")

from utils.api import PRN


class Texture_Generator:
    def __init__(self):
        prefix = os.getcwd()
        self.prn = PRN(is_dlib=True, prefix=prefix[:-6])

    def get_texture(self, image, seg):
        pos = self.prn.process(image)
        image = image
        face_texture = cv2.remap(
            image,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        seg_texture = cv2.remap(
            seg,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        del pos
        return face_texture, seg_texture
