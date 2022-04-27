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
from function import calc_mean_std
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
        
class StyleFolderDataset(data.Dataset):
    def __init__(self, jsonPath, imageLocation, transform=None):
        super(StyleFolderDataset, self).__init__()
        with open(jsonPath) as f:
            jsonData = json.load(f)
        
        self.paths = []
        self.sentences = []
        
        for image in jsonData['images']:
            self.paths.append(os.path.join(imageLocation, image['filename']))
            positive_sentences = []
            negative_sentences = []
            sentence_sentiments = []
            for sentence in image['sentences']:
                sentence_sentiments.append(sentence['sentiment'])
                if sentence['sentiment'] == 0:
                    negative_sentences.append(sentence['raw'])
                else:
                    positive_sentences.append(sentence['raw'])
            if sum(sentence_sentiments)/len(sentence_sentiments) > 0.5:
                self.sentences.append(positive_sentences)
            else:
                self.sentences.append(negative_sentences)
        
        self.transform = transform
  
    def __getitem__(self, index):
        
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        sentences = self.sentences[index]
        sentence = random.choice(sentences)
        return img, sentence

    def __len__(self):
        return len(self.paths)
      
    def name(self):
        return 'StyleFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--style_json', type=str, default='./senticap_dataset.json',
                    help='Path to JSON file for the Senticap dataset')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='./decoder.pth')
# training options
parser.add_argument('--save_dir', default='./experiments_generator',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--tv_weight', type=float, default=0.0)
parser.add_argument('--realism_weight', type=float, default=0.0)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--virtual_batch_mult', type=int, default=4)
args = parser.parse_args()

if torch.cuda.is_available() and args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
decoder.load_state_dict(torch.load(args.decoder))
decoder.eval()

vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

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
            nn.Conv2d(768, 768, 3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(768, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
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
    def __init__(self, enc_type, hidden_dim, output_dim, device):
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
        

text_encoder = TextEncoder('sentence-transformers/all-mpnet-base-v2', 1024, 1024, device)           

network = net.NetPSGen(vgg, decoder, text_encoder, device)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = StyleFolderDataset(args.style_json, args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.AdamW(network.decoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10*args.lr, total_steps=args.max_iter)

style_losses = []
content_losses = []
tv_losses = []
realism_losses = []
direct_losses = []

for i in tqdm(range(args.max_iter)):
    gc.collect()
    #adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images, style_texts = next(style_iter)
    style_images = style_images.to(device)
    loss_direct, loss_c, loss_s, loss_tv, loss_real = network(content_images, style_images, style_texts)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    if loss_tv is not None:
        loss_tv = args.tv_weight * loss_tv
    else:
        loss_tv = torch.tensor(0.)
    if loss_real is not None:
        loss_real = args.realism_weight * loss_real
    else:
        loss_real = torch.tensor(0.)
    loss = loss_direct + loss_c + loss_s + loss_tv + loss_real
    loss.backward()

    if (i + 1) % args.virtual_batch_mult == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_total_variation', loss_tv.item(), i + 1)
    writer.add_scalar('loss_photorealism', loss_real.item(), i + 1)

    style_losses.append(loss_s.item())
    content_losses.append(loss_c.item())
    tv_losses.append(loss_tv.item())
    realism_losses.append(loss_real.item())
    direct_losses.append(loss_direct.item())
    
    if (i + 1) % args.log_interval == 0:
        print('Direct Loss: ', sum(direct_losses)/len(direct_losses), 'Content Loss:', sum(content_losses)/len(content_losses), ', Style Loss:', sum(style_losses)/len(style_losses), ', Variation Loss:', sum(tv_losses)/len(tv_losses), ', Photorealism Loss:', sum(realism_losses)/len(realism_losses))
        style_losses = []
        content_losses = []
        tv_losses = []
        realism_losses = []
              
    
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        torch.save(network, save_dir /
                   'iter_{:d}.pth'.format(i + 1))

writer.close()
