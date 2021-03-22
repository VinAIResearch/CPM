import os
from parser import get_args

import cv2
import numpy as np
import torch
from dataloader import Dataset
from models import Segmentor
from torch.utils.data import DataLoader
from ultils import get_preprocessing


# path = '/home/ubuntu/segmentation4makeup/checkpoints/resnet50/best_model.pth'
print("Hi, ≧◡≦, parsing arguments...")
args = get_args()

# load best saved checkpoint
best_model_path = os.path.join(args.output_path, "best_model.pth")
best_model = Segmentor(args)
best_model.test_model(best_model_path)
print("Loaded model from: ", best_model_path)
# create test dataset
test_dataset = Dataset(
    args.x_test_dir,
    args.y_test_dir,
    #                        augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(best_model.preprocessing_fn),
    classes=args.classes,
)

test_dataloader = DataLoader(test_dataset)

logs = best_model.test_model.run(test_dataloader)

# --- Visualize some images
test_dataset_vis = Dataset(
    args.x_test_dir,
    args.y_test_dir,
    classes=args.classes,
)

if not os.path.isdir(os.path.join(args.output_path, "test")):
    os.mkdir(os.path.join(args.output_path, "test"))
    print("Mkdirs ", os.path.join(args.output_path, "test"))

for i in range(10):
    n = np.random.choice(len(test_dataset))
    image_vis = test_dataset_vis[n][0].astype("uint8")
    image, gt_mask = test_dataset[n]
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(args.device).unsqueeze(0)
    pr_mask = best_model.model.predict(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().numpy().round()

    gt_mask = np.stack([gt_mask] * 3, axis=2)
    pr_mask = np.stack([pr_mask] * 3, axis=2)

    output = np.concatenate([image_vis, gt_mask * 150, pr_mask * 150], axis=1)
    #     output = np.moveaxis(output, 0, -1)
    cv2.imwrite(os.path.join(args.output_path, "{}.png".format(i)), output)
