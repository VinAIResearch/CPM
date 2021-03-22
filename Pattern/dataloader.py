import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = [0]
    # CLASSES = ['background', 'face', 'righteyebrow', 'lefteyebrow', 'righteye', 'lefteye',
    #            'nose', 'upperlip', 'lowerlip', 'hair', 'rightear', 'leftear', 'neck', 'none', 'sticker']

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print("Got   : {:>15}".format(len(classes)))
        print("Name  : {:>15}".format(str(classes)))

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        # import pdb
        # pdb.set_trace()
        image = np.array(Image.open(self.images_fps[i]))
        mask = np.array(Image.open(self.masks_fps[i]))
        mask = [mask[:, :, 1] / 255]
        mask = np.stack(mask, axis=-1).astype("float")

        # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)
