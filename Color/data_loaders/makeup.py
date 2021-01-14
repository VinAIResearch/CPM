import os, glob, cv2
import torch
import random
import linecache
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class MAKEUP(Dataset):
    def __init__(self, image_path, transform, mode, transform_mask, cls_list):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.transform_mask = transform_mask

        # self.cls_list = cls_list
        # self.cls_A = cls_list[0]
        # self.cls_B = cls_list[1]

        # for cls in self.cls_list:
        #     setattr(self, "train_" + cls + "_list_path", os.path.join(self.image_path, "train_" + cls + ".txt"))
        #     setattr(self, "train_" + cls + "_lines", open(getattr(self, "train_" + cls + "_list_path"), 'r').readlines())
        #     setattr(self, "num_of_train_" + cls + "_data", len(getattr(self, "train_" + cls + "_lines")))
        # for cls in self.cls_list:
        #     if self.mode == "test_all":
        #         setattr(self, "test_" + cls + "_list_path", os.path.join(self.image_path, "test_" + cls + "_all.txt"))
        #         setattr(self, "test_" + cls + "_lines", open(getattr(self, "test_" + cls + "_list_path"), 'r').readlines())
        #         setattr(self, "num_of_test_" + cls + "_data", len(getattr(self, "test_" + cls + "_lines")))
        #     else:
        #         setattr(self, "test_" + cls + "_list_path", os.path.join(self.image_path, "test_" + cls + ".txt"))
        #         setattr(self, "test_" + cls + "_lines", open(getattr(self, "test_" + cls + "_list_path"), 'r').readlines())
        #         setattr(self, "num_of_test_" + cls + "_data", len(getattr(self, "test_" + cls + "_lines")))
        self.A_seg = glob.glob(os.path.join(image_path, 'segs', 'non-makeup', '*.png'))
        self.As = [os.path.join(image_path, 'imgs', 'non-makeup', x.split('/')[-1]) for x in self.A_seg]
        self.B_seg = glob.glob(os.path.join(image_path, 'segs', 'makeup', '*.png'))
        self.Bs = [os.path.join(image_path, 'imgs', 'makeup', x.split('/')[-1]) for x in self.B_seg]
        self.noiA = len(self.As)
        self.noiB = len(self.Bs)
        print(self.noiA, self.noiB)
        # print ('Start preprocessing dataset..!')
        # self.preprocess()
        # print ('Finished preprocessing dataset..!')

    # def preprocess(self):
        # for cls in self.cls_list:
        #     setattr(self, "train_" + cls + "_filenames", [])
        #     setattr(self, "train_" + cls + "_mask_filenames", [])

        #     lines = getattr(self, "train_" + cls + "_lines")
        #     random.shuffle(lines)

        #     for i, line in enumerate(lines):
        #         splits = line.split()
        #         getattr(self, "train_" + cls + "_filenames").append(splits[0])
        #         getattr(self, "train_" + cls + "_mask_filenames").append(splits[1])

        # for cls in self.cls_list:
        #     setattr(self, "test_" + cls + "_filenames", [])
        #     setattr(self, "test_" + cls + "_mask_filenames", [])
        #     lines = getattr(self, "test_" + cls + "_lines")
        #     for i, line in enumerate(lines):
        #         splits = line.split()
        #         getattr(self, "test_" + cls + "_filenames").append(splits[0])
        #         getattr(self, "test_" + cls + "_mask_filenames").append(splits[1])

        # if self.mode == "test_baseline":
        #     setattr(self, "test_" + self.cls_A + "_filenames", os.listdir(os.path.join(self.image_path, "baseline", "org_aligned")))
        #     setattr(self, "num_of_test_" + self.cls_A + "_data", len(os.listdir(os.path.join(self.image_path, "baseline", "org_aligned"))))
        #     setattr(self, "test_" + self.cls_B + "_filenames", os.listdir(os.path.join(self.image_path, "baseline", "ref_aligned")))
        #     setattr(self, "num_of_test_" + self.cls_B + "_data", len(os.listdir(os.path.join(self.image_path, "baseline", "ref_aligned"))))

    # def __getitem__(self, index):
    #     if self.mode == 'train' or self.mode == 'train_finetune':
    #         index_A = random.randint(0, getattr(self, "num_of_train_" + self.cls_A + "_data") - 1)
    #         index_B = random.randint(0, getattr(self, "num_of_train_" + self.cls_B + "_data") - 1)
    #         image_A = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_A + "_filenames")[index_A])).convert("RGB")
    #         image_B = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_B + "_filenames")[index_B])).convert("RGB")
    #         mask_A = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_A + "_mask_filenames")[index_A]))
    #         mask_B = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_B + "_mask_filenames")[index_B]))
    #         return self.transform(image_A), self.transform(image_B), self.transform_mask(mask_A), self.transform_mask(mask_B)
    #     if self.mode in ['test', 'test_all']:
    #         #"""
    #         image_A = Image.open(os.path.join(self.image_path, getattr(self, "test_" + self.cls_A + "_filenames")[index // getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
    #         image_B = Image.open(os.path.join(self.image_path, getattr(self, "test_" + self.cls_B + "_filenames")[index % getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
    #         return self.transform(image_A), self.transform(image_B)
    #     if self.mode == "test_baseline":
    #         image_A = Image.open(os.path.join(self.image_path, "baseline", "org_aligned", getattr(self, "test_" + self.cls_A + "_filenames")[index // getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
    #         image_B = Image.open(os.path.join(self.image_path, "baseline", "ref_aligned", getattr(self, "test_" + self.cls_B + "_filenames")[index % getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
    #         return self.transform(image_A), self.transform(image_B)
    def __getitem__(self, index):
        if self.mode == 'train':
            idxA = random.choice(range(self.noiA))
            idxB = random.choice(range(self.noiB))

            mask_A = Image.open(self.A_seg[idxA]).convert("RGB")
            mask_B = Image.open(self.B_seg[idxB]).convert("RGB")
            
            image_A = Image.open(self.As[idxA]).convert("RGB")
            image_B = Image.open(self.Bs[idxB]).convert("RGB")
            
            image_A = Image.fromarray(cv2.resize(np.array(image_A), (256, 256)))
            image_B = Image.fromarray(cv2.resize(np.array(image_B), (256, 256)))
            return self.transform(image_A), self.transform(image_B), self.transform_mask(mask_A), self.transform_mask(mask_B)

    def __len__(self):
        if self.mode == 'train' or self.mode == 'train_finetune':
            num_A = len(self.As)
            num_B = len(self.Bs)
            return max(num_A, num_B)
        
        elif self.mode in ['test', "test_baseline", 'test_all']:
            num_A = len(self.As)
            num_B = len(self.Bs)
            return num_A * num_B

    # def __len__(self):
        # if self.mode == 'train' or self.mode == 'train_finetune':
        #     num_A = getattr(self, 'num_of_train_' + self.cls_list[0] + '_data')
        #     num_B = getattr(self, 'num_of_train_' + self.cls_list[1] + '_data')
        #     return max(num_A, num_B)
        # elif self.mode in ['test', "test_baseline", 'test_all']:
        #     num_A = getattr(self, 'num_of_test_' + self.cls_list[0] + '_data')
        #     num_B = getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')
        #     return num_A * num_B
