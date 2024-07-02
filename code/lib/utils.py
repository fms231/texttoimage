import os
import errno
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from PIL import Image

import importlib
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# 选择模型
def choose_model(model):
    '''choose models
    '''
    model = importlib.import_module(".%s"%(model), "models")
    NetG, NetD, NetC, CLIP_IMG_ENCODER, CLIP_TXT_ENCODER = model.NetG, model.NetD, model.NetC, model.CLIP_IMG_ENCODER, model.CLIP_TXT_ENCODER
    return NetG,NetD,NetC,CLIP_IMG_ENCODER, CLIP_TXT_ENCODER

# 计算参数数量
def params_count(model):
    model_size = np.sum([p.numel() for p in model.parameters()]).item()
    return model_size

# 制作目录
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
# 加载FID所用参数
def load_npz(path):
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s

# config
def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg

def str2bool_dict(dict):
    for key,value in dict.items():
        if type(value)==str:
            if value.lower() in ('yes','true'):
                dict[key] = True
            elif value.lower() in ('no','false'):
                dict[key] = False
            else:
                None
    return dict

# 合并参数
def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        args.update(opt)
        args = str2bool_dict(args)
        args = edict(args)
    return args

# 加载优化器权重
def load_opt_weights(optimizer, weights):
    optimizer.load_state_dict(weights)
    return optimizer

# 加载模型和优化器权重
def load_models_opt(netG, netD, netC, optim_G, optim_D, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'])
    netD = load_model_weights(netD, checkpoint['model']['netD'])
    netC = load_model_weights(netC, checkpoint['model']['netC'])
    optim_G = load_opt_weights(optim_G, checkpoint['optimizers']['optimizer_G'])
    optim_D = load_opt_weights(optim_D, checkpoint['optimizers']['optimizer_D'])
    return netG, netD, netC, optim_G, optim_D

# 加载模型
def load_models(netG, netD, netC, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'])
    netD = load_model_weights(netD, checkpoint['model']['netD'])
    netC = load_model_weights(netC, checkpoint['model']['netC'])
    return netG, netD, netC

# 加载生成器
def load_netG(netG, path, train):
    checkpoint = torch.load(path, map_location="cpu")
    netG = load_model_weights(netG, checkpoint['model']['netG'], train)
    return netG

# 加载模型权重
def load_model_weights(model, weights, train=True):
    state_dict = weights
    model.load_state_dict(state_dict)
    return model

# 保存模型权重
def save_models_opt(netG, netD, netC, optG, optD, epoch, save_path):
    state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
            'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
            'epoch': epoch}
    torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))

# 保存模型
def save_models(netG, netD, netC, epoch, save_path):
    state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}}
    torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))

# 保存模型和优化器
def save_checkpoints(netG, netD, netC, optG, optD, scaler_G, scaler_D, epoch, save_path):
    state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
            'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
            "scalers": {"scaler_G": scaler_G.state_dict(), "scaler_D": scaler_D.state_dict()},\
            'epoch': epoch}
    torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))

# 转化为CLIP输入
def transf_to_CLIP_input(inputs):
    device = inputs.device
    if len(inputs.size()) != 4:
        raise ValueError('Expect the (B, C, X, Y) tensor.')
    else:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
        inputs = ((inputs+1)*0.5-mean)/var
        return inputs.float()
