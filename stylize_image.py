import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import os
import random
import json
from sentence_transformers import models
import gc
gc.enable()

import net
from net import NetPS
from sampler import InfiniteSamplerWrapper
from PIL import Image

parser = argparse.ArgumentParser()
# Basic options

parser.add_argument('--content_image', type=str)
parser.add_argument('--style_image', type=str)
parser.add_argument('--decoder_weight', type=str)
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

if torch.cuda.is_available() and args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

decoder = torch.load(args.decoder_weight, map_location=device)
decoder.eval()

#decoder = net.decoder
#decoder.load_state_dict(torch.load(args.decoder_weight))
#decoder.eval()

vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder, device)
network.eval()
#network.change_device(device)

def transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

img_transform = transform()

content_image = img_transform(Image.open(args.content_image)).unsqueeze(0)
style_image = img_transform(Image.open(args.style_image)).unsqueeze(0)

stylized_im = network.stylize(content_image, style_image)

stylized_im_pil = transforms.ToPILImage()(stylized_im[0])
stylized_im_pil = stylized_im_pil.save('stylized.png')

content_im_pil = transforms.ToPILImage()(content_image[0])
content_im_pil = content_im_pil.save('content.png')

style_im_pil = transforms.ToPILImage()(style_image[0])
style_im_pil = style_im_pil.save('style.png')