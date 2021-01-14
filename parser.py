import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_alpha', default = './pretrained_seg/alpha.pth', type=str)
#     parser.add_argument('--checkpoints_skin', default = '/home/ubuntu/pretrained_seg/1alpha.pth', type=str)
    parser.add_argument('--checkpoints_mkup', default = './G.pth', type=str)
    
    parser.add_argument('--output_path', default = '/home/ubuntu/checkpoints', type = str)
    
    parser.add_argument('--device', default = 'cuda', type = str)
    parser.add_argument('--classes', nargs='+', type=str, default = [1])
    
    parser.add_argument('--activation', default = 'sigmoid', type = str)
    parser.add_argument('--encoder', default = 'resnet50', type=str)
    parser.add_argument('--decoder', default = 'unet', type=str)
    parser.add_argument('--encoder_weights', default = 'imagenet', type = str)
    
    parser.add_argument('--batch_size', default = '1', type = int)
    parser.add_argument('--prn', default = True, type=bool)
    parser.add_argument('--mkup', default = True, action='store_false')
    
    parser.add_argument('--path', type=str, default = '/vinai/thaontp79/cvpr2021/imgs4vis/itw')
    parser.add_argument('--savedir', type=str, default = '/vinai/thaontp79/cvpr2021/qualitative/my/itw')

    args = parser.parse_args()
    
    print('           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰')
    print('🎵 hhey, arguments are here if you need to check 🎵')
    for arg in vars(args):
        print('{:>15}: {:>30}'.format(str(arg), str(getattr(args, arg))))
    print()
    return args

if __name__ == "__main__":
    get_args()
    