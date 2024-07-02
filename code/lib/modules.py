from scipy import linalg
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.utils import transf_to_CLIP_input
from lib.datasets import prepare_data
from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
# import torch.distributed as dist


############   GAN   ############
# 训练
def train(dataloader, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, args):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC, image_encoder = netG.train(), netD.train(), netC.train(), image_encoder.train()
    loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        ##############
        # Train D  
        ##############
        optimizerD.zero_grad()
        # prepare_data
        real, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
        real = real.requires_grad_()
        sent_emb = sent_emb.requires_grad_()
        words_embs = words_embs.requires_grad_()
        # predict real
        CLIP_real,real_emb = image_encoder(real)
        real_feats = netD(CLIP_real)
        pred_real, errD_real = predict_loss(netC, real_feats, sent_emb, negtive=False)
        # predict mismatch
        mis_sent_emb = torch.cat((sent_emb[1:], sent_emb[0:1]), dim=0).detach()
        _, errD_mis = predict_loss(netC, real_feats, mis_sent_emb, negtive=True)
        # synthesize fake images
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = netG(noise, sent_emb)
        CLIP_fake, fake_emb = image_encoder(fake)
        fake_feats = netD(CLIP_fake.detach())
        _, errD_fake = predict_loss(netC, fake_feats, sent_emb, negtive=True)
        # MA-GP
        errD_MAGP = MA_GP_FP32(CLIP_real, sent_emb, pred_real)
        # whole D loss
        errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP
        # update D
        errD.backward()
        optimizerD.step()
        ##############
        # Train G  
        ##############
        optimizerG.zero_grad()
        fake_feats = netD(CLIP_fake)
        output = netC(fake_feats, sent_emb)
        text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
        errG = -output.mean() - args.sim_w*text_img_sim
        errG.backward()
        optimizerG.step()
        # update loop information
        loop.update(1)
        loop.set_description(f'Train Epoch [{epoch}/{max_epoch}]')
        loop.set_postfix()
    loop.close()

# 测试
def test(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    FID, TI_sim = calculate_FID_CLIP_sim(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size)
    return FID, TI_sim

# 保存模型
def save_model(netG, netD, netC, optG, optD, epoch, step, save_path):
    state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
            'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
            'epoch': epoch}
    torch.save(state, '%s/state_epoch_%03d_%03d.pth' % (save_path, epoch, step))


#########   MAGP   ########
def MA_GP_FP32(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp

# 计算FID和CLIP相似度
def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    """ Calculates the FID """
    clip_cos = torch.FloatTensor([0.0]).to(device)
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    # n_gpu = dist.get_world_size()
    n_gpu = 1
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))

    loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb,eval=True).float()
                # norm_ip(fake_imgs, -1, 1)
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                for i in range(len(output)):
                    output[i] = pred
                pred_all = torch.cat(output,dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().numpy()
            # update loop information
            loop.update(1)
            if epoch==-1:
                loop.set_description('Evaluating]')
            else:
                loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    loop.close()
    # CLIP-score
    CLIP_score_gather = list(torch.empty_like(clip_cos) for _ in range(n_gpu))
    # dist.all_gather(CLIP_score_gather, clip_cos)
    for i in range(len(CLIP_score_gather)):
        CLIP_score_gather[i] = clip_cos
    clip_score = torch.cat(CLIP_score_gather, dim=0).mean().item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value,clip_score

# 计算clip相似度
def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim

# 计算loss
def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err

# 计算hinge loss
def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err

# 计算frechet distance  FID
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
