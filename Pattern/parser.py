import argparse
import glob
import os


def count_images(args):
    if os.path.isdir(args.datapath):
        x_train_dir = os.path.join(args.datapath, "train/img")
        y_train_dir = os.path.join(args.datapath, "train/seg")

        x_test_dir = os.path.join(args.datapath, "test/img")
        y_test_dir = os.path.join(args.datapath, "test/seg")
        if not args.test:
            print("           ─── ･ ｡ﾟ☆: *.☽ .* :☆ﾟ. ───")

            print(
                "--- Train: {:>6} imgs, together with {:>6} masks".format(
                    len(glob.glob(os.path.join(x_train_dir, "*.png"))),
                    len(glob.glob(os.path.join(y_train_dir, "*.png"))),
                )
            )
            print(
                "     Test: {:>6} imgs, together with {:>6} masks".format(
                    len(glob.glob(os.path.join(x_test_dir, "*.png"))),
                    len(glob.glob(os.path.join(y_test_dir, "*.png"))),
                )
            )
            print("           ─── ･ ｡ﾟ☆: *.☽ .* :☆ﾟ. ───")
            print()
        else:
            print("           ─── ･ ｡ﾟ☆: *.☽ .* :☆ﾟ. ───")
            print(
                "    Test: {:>6} imgs, together with {:>6} masks".format(
                    len(glob.glob(os.path.join(x_test_dir, "*.png"))),
                    len(glob.glob(os.path.join(y_test_dir, "*.png"))),
                )
            )
            print("           ─── ･ ｡ﾟ☆: *.☽ .* :☆ﾟ. ───")
            print()
        return x_train_dir, y_train_dir, x_test_dir, y_test_dir
    else:
        raise Exception("Directory {} is not valid".format(args.datapath))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="/home/ubuntu/synthesis_data", type=str)
    parser.add_argument("--output_path", default="../../checkpoints", type=str)
    parser.add_argument("--logdir", default="./runs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--classes", nargs="+", type=str, default=[0])

    parser.add_argument("--activation", default="sigmoid", type=str)
    parser.add_argument("--encoder", default="resnet50", type=str)
    parser.add_argument("--decoder", default="unet", type=str)
    parser.add_argument("--encoder_weights", default="imagenet", type=str)

    parser.add_argument("--batch_size", default="16", type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--visual_gap", default=5, type=int)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    print("🎵 hhey, arguments are here if you need to check 🎵")
    for arg in vars(args):
        print("{:>15}: {:>30}".format(str(arg), str(getattr(args, arg))))
    print()

    args.x_train_dir, args.y_train_dir, args.x_test_dir, args.y_test_dir = count_images(args)

    args.output_path = os.path.join(args.output_path, str(args.encoder) + "_" + str(args.decoder))
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)
    return args


if __name__ == "__main__":
    get_args()
