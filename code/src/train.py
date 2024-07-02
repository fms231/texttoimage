import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np

import torch
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,merge_args_yaml
from lib.utils import load_models_opt,save_models_opt,load_npz,params_count
from lib.perpare import prepare_dataloaders,prepare_models
from lib.modules import test as test, train as train

# 接受参数
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/birds.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
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

# 主函数
def main(args):
    # 模型保存路径
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.CONFIG_NAME))
    mkdir_p(args.model_save_file)
    # 加载训练集
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    # 加载模型
    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    # 显示模型参数
    print('**************G_paras: ',params_count(netG))
    print('**************D_paras: ',params_count(netD)+params_count(netC))
    # 优化器
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=args.lr_d, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    # 用于计算FID
    m1, s1 = load_npz(args.npz_path)
    # 开始轮次
    start_epoch = 1
    # 有预训练模型就加载
    if args.state_epoch!=1:
        # 开始继续训练
        start_epoch = args.state_epoch + 1
        # 模型路径
        path = osp.join(args.pretrained_model_path, 'state_epoch_%03d.pth'%(args.state_epoch))
        # 加载模型
        netG, netD, netC, optimizerG, optimizerD = load_models_opt(netG, netD, netC, optimizerG, optimizerD, path)
        print("load successful")
    print("Start Training")
    # Start training
    # 测试和保存间隔
    test_interval,save_interval = args.test_interval,args.save_interval
    #torch.cuda.empty_cache()
    # start_epoch = 1
    # 训练
    for epoch in range(start_epoch, args.max_epoch, 1):
        # 计时
        start_t = time.time()
        # training
        args.current_epoch = epoch
        torch.cuda.empty_cache()
        train(train_dl, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, args)
        torch.cuda.empty_cache()
        # save
        if epoch%save_interval==0:
            save_models_opt(netG, netD, netC, optimizerG, optimizerD, epoch, args.model_save_file)
            print('save successful')
            torch.cuda.empty_cache()
        # test
        if epoch%test_interval==0:
            #torch.cuda.empty_cache()
            FID, TI_score = test(valid_dl, text_encoder, netG, CLIP4evl, args.device, m1, s1, epoch, args.max_epoch, args.sample_times, args.z_dim, args.batch_size)
            print('FID: %.2f, CLIP_Score: %.2f' % (FID, TI_score*100))
            torch.cuda.empty_cache()
        end_t = time.time()
        print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
        print('*'*40)



if __name__ == "__main__":
    # 获取超参数
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    # gpu/cpu
    if args.cuda:
        torch.cuda.manual_seed_all(args.manual_seed)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)

