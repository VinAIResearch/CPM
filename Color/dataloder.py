from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data_loaders.makeup import MAKEUP
import torch
import numpy as np
import PIL

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def get_loader(data_config, config, mode="train"):
    # return the DataLoader
    dataset_name = data_config.name
    transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=PIL.Image.NEAREST),
        ToTensor])
    print(config.data_path)
    #"""
    if mode=="train":
        dataset_train = eval(dataset_name)(data_config.dataset_path, transform=transform, mode= "train",\
                                                            transform_mask=transform_mask, cls_list = config.cls_list)
        dataset_test = eval(dataset_name)(data_config.dataset_path, transform=transform, mode= "test",\
                                                                transform_mask=transform_mask, cls_list = config.cls_list)

        #"""
        data_loader_train = DataLoader(dataset=dataset_train,
                                batch_size=config.batch_size,
                                shuffle=True)
    
    if mode=="test":
        data_loader_train = None
        dataset_test = eval(dataset_name)(data_config.dataset_path, transform=transform, mode= "test",\
                                                            transform_mask =transform_mask, cls_list = config.cls_list)




    data_loader_test = DataLoader(dataset=dataset_test,
                             batch_size=1,
                             shuffle=False)

    return [data_loader_train, data_loader_test]