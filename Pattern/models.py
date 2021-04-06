import segmentation_models_pytorch as smp
import torch


class Segmentor:
    def __init__(self, args):
        # create segmentation model with pretrained encoder
        if args.decoder == "fpn":
            self.model = smp.FPN(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        elif args.decoder == "unet":
            self.model = smp.Unet(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        elif args.decoder == "deeplabv3":
            self.model = smp.DeepLabV3(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        else:
            self.model = smp.PSPNet(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Accuracy(threshold=0.5),
            smp.utils.metrics.Precision(),
            smp.utils.metrics.Recall(),
        ]

        self.optimizer = torch.optim.Adam(
            [
                dict(params=self.model.parameters(), lr=0.0001),
            ]
        )

        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=args.device,
            verbose=True,
        )
        #         return train_epoch
        self.valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=args.device,
            verbose=True,
        )

    def test_model(self, path):
        self.test_model = smp.utils.train.ValidEpoch(
            torch.load(path),
            loss=self.loss,
            metrics=self.metrics,
            device="cuda",
            verbose=True,
        )
        self.model = torch.load(path)
