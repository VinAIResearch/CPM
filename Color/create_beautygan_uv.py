"""
Generate texture for dataset_folder
"""

import argparse
import glob
import os

import cv2
from texture_generator import Texture_Generator
from tqdm import tqdm as tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", type=str)
    parser.add_argument("--savedir", default="./MakeupTransfer_UV", type=str)
    args = parser.parse_args()
    print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    for arg in vars(args):
        print("{:>15}: {:>30}".format(str(arg), str(getattr(args, arg))))
    print()
    return args


if __name__ == "__main__":
    args = get_args()
    generator = Texture_Generator()

    list_imgs = glob.glob(os.path.join(args.path, "images", "*", "*.png"))
    filenames = [x.split("/all/images/")[-1] for x in list_imgs]
    list_segs = [os.path.join(args.path, "segs", x) for x in filenames]

    print("Found {} images, together with {} segmentation mask".format(len(list_imgs), len(list_segs)))

    subdirs = [
        os.path.join(args.savedir, x) for x in ["images/non-makeup", "images/makeup", "segs/non-makeup", "segs/makeup"]
    ]
    for subpath in subdirs:
        if not os.path.isdir(subpath):
            os.makedirs(subpath)
            print("Created: ", subpath)

    print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    print("")
    print("New images will be saved in: ", args.savedir)

    for i in tqdm(range(0, len(list_imgs))):
        image = cv2.imread(list_imgs[i])
        seg = cv2.imread(list_segs[i])
        uv_texture, uv_seg = generator.get_texture(image, seg)

        cv2.imwrite(os.path.join(args.savedir, "images", filenames[i]), uv_texture)
        cv2.imwrite(os.path.join(args.savedir, "segs", filenames[i]), uv_seg)
