import torch
import torch.nn as nn

from function import adaptive_instance_normalization as style_adain
from function import text_based_adaptive_instance_normalization as text_adain
from function import calc_mean_std

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

from sentence_transformers import SentenceTransformer, models
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from torch.autograd import Variable
import lpips

def batch_to_device(batch, target_device: torch.device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class SentenceTransformerv2(SentenceTransformer):
    def __init__(self, modules = None):
        super().__init__(modules=modules)
    def encode_with_gradients(self, sentences: Union[str, List[str]],
               device: str = None,
               normalize_embeddings: bool = False):

        if device is None:
            device = self._target_device

        self.to(device)
        
        features = self.tokenize(sentences)
        features = batch_to_device(features, device)
        out_features = self.forward(features)
        embeddings = out_features['sentence_embedding']
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

class TVLoss(nn.Module):

    def __init__(self, device):
        super(TVLoss, self).__init__()
        self.ky = np.array([
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]]
        ])
        self.kx = np.array([
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]]
        ])
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(torch.from_numpy(self.kx).float().unsqueeze(0).to(device),
                                          requires_grad=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(self.ky).float().unsqueeze(0).to(device),
                                          requires_grad=False)

    def forward(self, input):
        height, width = input.size()[2:4]
        gx = self.conv_x(input)
        gy = self.conv_y(input)

        return torch.sum(gx**2 + gy**2)/2.0
        
class LPIPSLoss(torch.nn.Module):
    def __init__(self, device):
        super(LPIPSLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return torch.mean(self.lpips_loss(input, target))


class RealLoss(nn.Module):
    
    def __init__(self, device):
        super(RealLoss, self).__init__()
        self.device = device
        
    def _rolling_block(self, A, block=(3, 3)):
        """Applies sliding window to given matrix."""
        shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
        strides = (A.strides[0], A.strides[1]) + A.strides
        return as_strided(A, shape=shape, strides=strides)
    
    
    def compute_laplacian(self, img, mask=None, eps=10 ** (-7), win_rad=1):
        """Computes Matting Laplacian for a given image.
        Args:
            img: 3-dim numpy matrix with input image
            mask: mask of pixels for which Laplacian will be computed.
                If not set Laplacian will be computed for all pixels.
            eps: regularization parameter controlling alpha smoothness
                from Eq. 12 of the original paper. Defaults to 1e-7.
            win_rad: radius of window used to build Matting Laplacian (i.e.
                radius of omega_k in Eq. 12).
        Returns: sparse matrix holding Matting Laplacian.
        """
    
        win_size = (win_rad * 2 + 1) ** 2
        h, w, d = img.shape
        # Number of window centre indices in h, w axes
        c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
        win_diam = win_rad * 2 + 1
    
        indsM = np.arange(h * w).reshape((h, w))
        ravelImg = img.reshape(h * w, d)
        win_inds = self._rolling_block(indsM, block=(win_diam, win_diam))
    
        win_inds = win_inds.reshape(c_h, c_w, win_size)
        if mask is not None:
            mask = cv2.dilate(
                mask.astype(np.uint8), np.ones((win_diam, win_diam), np.uint8)
            ).astype(np.bool)
            win_mask = np.sum(mask.ravel()[win_inds], axis=2)
            win_inds = win_inds[win_mask > 0, :]
        else:
            win_inds = win_inds.reshape(-1, win_size)
    
        winI = ravelImg[win_inds]
    
        win_mu = np.mean(winI, axis=1, keepdims=True)
        win_var = np.einsum("...ji,...jk ->...ik", winI, winI) / win_size - np.einsum(
            "...ji,...jk ->...ik", win_mu, win_mu
        )
    
        inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))
    
        X = np.einsum("...ij,...jk->...ik", winI - win_mu, inv)
        vals = np.eye(win_size) - (1.0 / win_size) * (
            1 + np.einsum("...ij,...kj->...ik", X, winI - win_mu)
        )
    
        nz_indsCol = np.tile(win_inds, win_size).ravel()
        nz_indsRow = np.repeat(win_inds, win_size).ravel()
        nz_indsVal = vals.ravel()
        L = scipy.sparse.coo_matrix(
            (nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w)
        )
        return L
        
    def compute_matting_laplacians(self, images, consts=None, epsilon=1e-5, window_radius=1):
        laplacians_t = []
        images_np = torch.moveaxis(images, 1, 3).cpu().numpy()
        for image in images_np:
            laplacian_t = self.compute_laplacian(image)
            laplacians_t.append(torch.tensor(laplacian_t))
    
        return torch.stack(laplacians_t, dim=0)

    def forward(self, output, content):
        channel, height, width = output.size()[1:4]
        loss = 0
        self.L = Variable(self.compute_matting_laplacian(content), requires_grad=False).to(self.device)
        for i in range(channel):
            temp = input[:, i, :, :]
            temp = temp.reshape(-1, 1, height*width)
            r = torch.bmm(self.L, temp.t())
            loss += torch.bmm(temp , r)
        return loss


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)
'''
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)
'''
class LaplacianRegularizer(nn.Module):
    def __init__(self):
        super(LaplacianRegularizer, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, f):
        loss = 0.
        for i in range(f.shape[2]):
            for j in range(f.shape[3]):
                up = max(i-1, 0)
                down = min(i+1, f.shape[2] - 1)
                left = max(j-1, 0)
                right = min(j+1, f.shape[3] - 1)
                term = f[:,:,i,j].view(f.shape[0], f.shape[1], 1, 1).\
                        expand(f.shape[0], f.shape[1], down - up+1, right-left+1)
                loss += self.mse_loss(term, f[:, :, up:down+1, left:right+1])
        return loss

class NetPSGen(nn.Module):
    def __init__(self, encoder, decoder, text_encoder, device, use_realism_loss=False, use_tv_loss=True):
        super(NetPSGen, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.text_encoder = text_encoder
        self.mse_loss = nn.MSELoss()
        self.use_tv_loss = use_tv_loss
        self.use_realism_loss = use_realism_loss
        if self.use_tv_loss:
            self.tv_loss = TVLoss(device)
        if self.use_realism_loss:
            self.realism_loss = LaplacianRegularizer()
        self.device = device
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)  # swapped ch and w*h, transpose share storage with original
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    
    def text_encode(self, input):
        return self.text_encoder.encode(input)
        #return self.text_encoder.encode_with_gradients(input)

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        #print(input.size(), target.size())
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def calc_tv_loss(self, input):
        return self.tv_loss(input)
    
    def calc_realism_loss(self, output, content):
        return self.realism_loss(output, content)

    def forward(self, content, style, style_text, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        text_feat = self.text_encode(style_text)

        #print(text_feat.shape, style_feats[-1].shape)
        t = style_adain(content_feat, text_feat)
        
        text_mean, text_std = calc_mean_std(text_feat)
        style_mean, style_std = calc_mean_std(style_feats[-1])
        loss_direct = self.mse_loss(text_mean, style_mean)
        loss_direct += self.mse_loss(text_std, style_std)
        O = self.gram_matrix(text_feat)
        S = self.gram_matrix(style_feats[-1])
        loss_direct += self.mse_loss(O, S)

        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t.clone().detach().requires_grad_(False))
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        '''
        for i in range(4):
            O = self.gram_matrix(g_t_feats[i])
            S = self.gram_matrix(style_feats[i])
            loss_s += self.mse_loss(O, S)
        '''
        if self.use_tv_loss:
            loss_tv = self.calc_tv_loss(g_t)
        else:
            loss_tv = None
        if self.use_realism_loss:
            loss_real = self.calc_realism_loss(g_t)
        else:
            loss_real = None
        return loss_direct, loss_c, loss_s, loss_tv, loss_real
    
    def stylize(self, content, style_text, alpha=1.0):
        with torch.no_grad():
            content_feat = self.encode(content)
            text_feat = self.text_encode(style_text)
            t = style_adain(content_feat, torch.tensor(text_feat))
            t = alpha * t + (1 - alpha) * content_feat
            g_t = self.decoder(t)
        for i in range(3):
            g_t[:, i, :, :] -= g_t[:, i, :, :].min()
            g_t[:, i, :, :] /= g_t[:, i, :, :].max()
        return g_t
    
    def change_device(self, device):
        self.device = device
        self.enc_1 = self.enc_1.to(device)
        self.enc_2 = self.enc_2.to(device)
        self.enc_3 = self.enc_3.to(device)
        self.enc_4 = self.enc_4.to(device)
        self.decoder = self.decoder.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder.device = device
        return True

class NetPS(nn.Module):
    def __init__(self, encoder, decoder, text_encoder, device, use_realism_loss=False, use_tv_loss=True):
        super(NetPS, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.text_encoder = text_encoder
        self.mse_loss = nn.MSELoss()
        self.use_tv_loss = use_tv_loss
        self.use_realism_loss = use_realism_loss
        if self.use_tv_loss:
            self.tv_loss = TVLoss(device)
        if self.use_realism_loss:
            self.realism_loss = LaplacianRegularizer()
        self.device = device
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
    
    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)  # swapped ch and w*h, transpose share storage with original
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    
    def text_encode(self, input):
        return self.text_encoder.encode(input)
        #return self.text_encoder.encode_with_gradients(input)

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        #print(input.size(), target.size())
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def calc_tv_loss(self, input):
        return self.tv_loss(input)
    
    def calc_realism_loss(self, output, content):
        return self.realism_loss(output, content)

    def forward(self, content, style, style_text, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        text_feat = self.text_encode(style_text)

        t = text_adain(content_feat, text_feat)
        
        style_mean, style_std = calc_mean_std(style_feats[-1])
        loss_direct = nn.L1Loss()(text_feat[:, :512], nn.Flatten()(style_mean))
        loss_direct += nn.L1Loss()(text_feat[:, 512:], nn.Flatten()(style_std))

        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t.clone().detach().requires_grad_(False))
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        '''
        for i in range(4):
            O = self.gram_matrix(g_t_feats[i])
            S = self.gram_matrix(style_feats[i])
            loss_s += self.mse_loss(O, S)
        '''
        if self.use_tv_loss:
            loss_tv = self.calc_tv_loss(g_t)
        else:
            loss_tv = None
        if self.use_realism_loss:
            loss_real = self.calc_realism_loss(g_t)
        else:
            loss_real = None
        return loss_direct, loss_c, loss_s, loss_tv, loss_real
    
    def stylize(self, content, style_text, alpha=1.0):
        with torch.no_grad():
            content_feat = self.encode(content)
            text_feat = self.text_encode(style_text)
            t = text_adain(content_feat, torch.tensor(text_feat))
            t = alpha * t + (1 - alpha) * content_feat
            g_t = self.decoder(t)
        for i in range(3):
            g_t[:, i, :, :] -= g_t[:, i, :, :].min()
            g_t[:, i, :, :] /= g_t[:, i, :, :].max()
        return g_t
    
    def change_device(self, device):
        self.device = device
        self.enc_1 = self.enc_1.to(device)
        self.enc_2 = self.enc_2.to(device)
        self.enc_3 = self.enc_3.to(device)
        self.enc_4 = self.enc_4.to(device)
        self.decoder = self.decoder.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder.device = device
        return True



class Net(nn.Module):
    def __init__(self, encoder, decoder, device, use_realism_loss=False, use_tv_loss=True):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.use_tv_loss = use_tv_loss
        self.use_realism_loss = use_realism_loss
        self.device = device
        if self.use_tv_loss:
            self.tv_loss = TVLoss(device)
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)  # swapped ch and w*h, transpose share storage with original
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = style_adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], content_feat)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
       
        for i in range(4):
            O = self.gram_matrix(g_t_feats[i])
            S = self.gram_matrix(style_feats[i])
            loss_s += self.mse_loss(O, S)
        
        if self.use_tv_loss:
            loss_tv = self.tv_loss(g_t)
        else:
            loss_tv = None
        return loss_c, loss_s, loss_tv
    
    def stylize(self, content, style, alpha=1.0):
        with torch.no_grad():
            content_feat = self.encode(content)
            style_feat = self.encode(style)
            t = style_adain(content_feat, style_feat)
            t = alpha * t + (1 - alpha) * content_feat
            g_t = self.decoder(t)
        for i in range(3):
            g_t[:, i, :, :] -= g_t[:, i, :, :].min()
            g_t[:, i, :, :] /= g_t[:, i, :, :].max()
        return g_t

    def change_device(self, device):
        self.device = device
        self.enc_1 = self.enc_1.to(device)
        self.enc_2 = self.enc_2.to(device)
        self.enc_3 = self.enc_3.to(device)
        self.enc_4 = self.enc_4.to(device)
        self.decoder = self.decoder.to(device)
        self.tv_loss = self.tv_loss.to(device)
        self.tv_loss.device = device
        return True

