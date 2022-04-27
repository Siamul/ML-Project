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

def style_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
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
parser.add_argument('--save_dir', default='./experiments_dense',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--tv_weight', type=float, default=5.0)
parser.add_argument('--realism_weight', type=float, default=10 ** 4)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
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



class TextEncoder(nn.Module):
    def __init__(self, enc_type, hidden_dim, output_dim, device, n_layers=5):
        super().__init__()
        self.encoder = net.SentenceTransformer(enc_type)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.n_layers = n_layers
        self.input_layer = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.input_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.hidden_layers = []
        for i in range(n_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                ) 
            )
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        ) 
        self.device = device
    
    def forward(self, input):
        embedding = self.encoder.encode(input)
        intermediate1 = self.input_layer(torch.tensor(embedding).to(self.device))
        intermediate2 = self.input_layer2(intermediate1)
        for i in range(self.n_layers):
            hidden_output = self.hidden_layers[i](torch.cat([intermediate1, intermediate2], dim=1))
            intermediate1 = intermediate2
            intermediate2 = hidden_output
        output = self.output_layer(hidden_output)
        return output
    
    def encode(self, input):
        embedding = self.encoder.encode(input)
        intermediate1 = self.input_layer(torch.tensor(embedding).to(self.device))
        intermediate2 = self.input_layer2(intermediate1)
        for i in range(self.n_layers):
            hidden_output = self.hidden_layers[i](torch.cat([intermediate1, intermediate2], dim=1))
            intermediate1 = intermediate2
            intermediate2 = hidden_output
        output = self.output_layer(hidden_output)
        return output
        

text_encoder = TextEncoder('sentence-transformers/all-mpnet-base-v2', 1024, 1024, device, n_layers=64)           

network = net.NetPS(vgg, decoder, text_encoder, device)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = style_transform()

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

optimizer = torch.optim.Adam(network.text_encoder.parameters(), lr=args.lr, amsgrad=True)

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

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
