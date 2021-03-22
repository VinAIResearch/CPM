import os
from parser import get_args

import segmentation_models_pytorch as smp
import torch
from dataloader import Dataset
from models import Segmentor
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import get_preprocessing


args = get_args()
segmentor = Segmentor(args)
preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
train_dataset = Dataset(
    args.x_train_dir,
    args.y_train_dir,
    #     augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=args.classes,
)

valid_dataset = Dataset(
    args.x_test_dir,
    args.y_test_dir,
    #     augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=args.classes,
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# train model
print()
print("--- T R A I N I N G P R O C E S S ˚‧º·( 。ᗒ ‸  ◕ ✿)")
print()
print("           ⛰️   Climbing on top... ⛰️")
print()

max_score = 0

# -- Tensorboard

writer = SummaryWriter(args.logdir)
print("Tensorboard will be saved in: ", args.logdir)

if not os.path.isdir(os.path.join(args.output_path, "visual")):
    os.mkdir(os.path.join(args.output_path, "visual"))

for i in range(0, args.epoch):
    print("\nEpoch: {}".format(i))
    train_logs = segmentor.train_epoch.run(train_loader)
    test_logs = segmentor.valid_epoch.run(valid_loader)

    writer.add_scalar("train/loss", train_logs["dice_loss"], i)
    writer.add_scalar("train/iou_score", train_logs["iou_score"], i)
    writer.add_scalar("train/accuracy", train_logs["accuracy"], i)
    writer.add_scalar("train/precision", train_logs["precision"], i)
    writer.add_scalar("train/recall", train_logs["recall"], i)
    writer.add_scalar("test/iou_score", test_logs["iou_score"], i)
    writer.add_scalar("test/accuracy", test_logs["accuracy"], i)
    writer.add_scalar("test/precision", test_logs["precision"], i)
    writer.add_scalar("test/recall", test_logs["recall"], i)

    # do something (save model, change lr, etc.)
    if max_score < test_logs["iou_score"]:
        max_score = test_logs["iou_score"]
        torch.save(segmentor.model, os.path.join(args.output_path, "best_model.pth"))
        print("Model saved! ✔️")

    if i % 25 == 0:
        torch.save(segmentor.model, os.path.join(args.output_path, "epoch_{}.pth".format(i)))
        segmentor.optimizer.param_groups[0]["lr"] = 1e-5
        print("Decrease decoder learning rate to 1e-5! 🔥")

print("Congrats, you are on top of the mountainn ˚‧º·( 。ᗒ ‸  ◕ ✿)")
