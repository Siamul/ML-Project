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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(768, 768, 4, 1, 0, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(True),
            nn.Conv2d(768, 768, 3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(768, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # state size. (ngf) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output

class TextEncoder(nn.Module):
    def __init__(self, enc_type, hidden_dim, output_dim, device, n_layers=5):
        super().__init__()
        self.encoder = net.SentenceTransformer(enc_type)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.generator = Generator()
        self.device = device
    
    def forward(self, input):
        embedding = self.encoder.encode(input)
        output = self.generator(torch.tensor(embedding).reshape(-1,768,1,1).to(self.device))
        return output
    
    def encode(self, input):
        embedding = self.encoder.encode(input)
        output = self.generator(torch.tensor(embedding).reshape(-1,768,1,1).to(self.device))
        return output

'''
class TextEncoder(nn.Module):
    def __init__(self, enc_type, hidden_dim, output_dim, device, n_layers=5):
        super().__init__()
        self.encoder = net.SentenceTransformer(enc_type)
        self.encoder.eval()
        self.n_layers = n_layers
        self.input_layer = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU()
        )
        self.hidden_layers = []
        for i in range(n_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ) 
            )
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.device = device
    
    def forward(self, input):
        with torch.no_grad():
            embedding = self.encoder.encode(input)
        intermediate = self.input_layer(torch.tensor(embedding).to(self.device))
        for i in range(self.n_layers+1):
            intermediate = self.hidden_layers[i](intermediate)
        output = self.output_layer(intermediate)
        return output
    
    def encode(self, input):
        with torch.no_grad():
            embedding = self.encoder.encode(input)
        intermediate = self.input_layer(torch.tensor(embedding).to(self.device))
        for i in range(self.n_layers):
            intermediate = self.hidden_layers[i](intermediate)
        output = self.output_layer(intermediate)
        return output
'''

parser = argparse.ArgumentParser()
# Basic options

parser.add_argument('--content_image', type=str)
parser.add_argument('--style_text', type=str)
parser.add_argument('--weight', type=str)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

if torch.cuda.is_available() and args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

network = torch.load(args.weight, map_location=device)
network.change_device(device)

def transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

img_transform = transform()

content_image = img_transform(Image.open(args.content_image)).unsqueeze(0)
style_text = [args.style_text]

stylized_im = network.stylize(content_image, style_text)

stylized_im_pil = transforms.ToPILImage()(stylized_im[0])
stylized_im_pil = stylized_im_pil.save('stylized_ps.png')

content_im_pil = transforms.ToPILImage()(content_image[0])
content_im_pil = content_im_pil.save('content.png')
