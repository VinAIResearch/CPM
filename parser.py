import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_pattern', default = './checkpoints/pattern.pth', type=str)
    parser.add_argument('--checkpoint_color', default = './checkpoints/color.pth', type=str)

    parser.add_argument('--output_path', default = '/home/ubuntu/checkpoints', type = str)
    parser.add_argument('--device', default = 'cuda', type = str)
    parser.add_argument('--classes', nargs='+', type=str, default = [1])
    
    parser.add_argument('--activation', default = 'sigmoid', type = str)
    parser.add_argument('--encoder', default = 'resnet50', type=str)
    parser.add_argument('--decoder', default = 'unet', type=str)
    parser.add_argument('--encoder_weights', default = 'imagenet', type = str)
    
    parser.add_argument('--batch_size', default = '1', type = int)
    parser.add_argument('--prn', default = True, type=bool)
    parser.add_argument('--color_only', default = False, action='store_true')
    parser.add_argument('--pattern_only', default = False, action='store_true')
    
    # parser.add_argument('--path', type=str, default = '.')
    parser.add_argument('--savedir', type=str, default = '.')
    parser.add_argument('--input', type=str, default = './imgs/non-makeup.png', help='Path to input image (non-makeup)')
    parser.add_argument('--style', type=str, default = './imgs/style-2.png', help='Path to style image (makeup style | reference image)')
    parser.add_argument('--alpha', type=float, default = 0.7, help='opacity of color makeup')

    args = parser.parse_args()
    
    print('           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰')
    print('🎵 hhey, arguments are here if you need to check 🎵')
    for arg in vars(args):
        print('{:>15}: {:>30}'.format(str(arg), str(getattr(args, arg))))
    print()
    return args

if __name__ == "__main__":
    get_args()
    