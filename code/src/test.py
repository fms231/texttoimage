import sys
import os.path as osp
import random
import argparse
import numpy as np

import torch
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,merge_args_yaml
from lib.utils import load_netG,load_npz,save_models
from lib.perpare import prepare_dataloaders
from lib.perpare import prepare_models
from lib.modules import test as test

# 接受参数
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/birds.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--pretrained_model_path', type=str, default='model',
                        help='the model for training')
    parser.add_argument('--model', type=str, default='GALIP',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=100,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args): 
    # Build and load the generator
    # prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    # 获取模型
    state_path = args.pretrained_model_path
    m1, s1 = load_npz(args.npz_path)
    netG = load_netG(netG, state_path, args.train)
    # 保存模型
    save_models(netG, netD, netC, 0, './tmp')

    netG.eval()
    # 测试
    FID, TI_score = test(valid_dl, text_encoder, netG, CLIP4evl, args.device, m1, s1, -1, -1, \
                    args.sample_times, args.z_dim, args.batch_size)
    # 输出结果
    print('FID: %.2f, CLIP_Score: %.2f' % (FID, TI_score*100))


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manual_seed)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)


