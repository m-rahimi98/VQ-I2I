import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
from dataset import dataset_pair
from torch.utils.data import DataLoader
import os

from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 

import argparse

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='0',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='/eva_data0/dataset/cityscapes',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='cityscapes',
                    help="dataset directory name",
                    type=str)
    
    parser.add_argument("--ne", default=512,
                    help="the number of embedding",
                    type=int)

    parser.add_argument("--ed", default=512,
                    help="embedding dimension",
                    type=int)

    parser.add_argument("--z_channel",default=128,
                    help="z channel",
                    type=int)
    

    parser.add_argument("--epoch_start", default=1,
                    help="start from",
                    type=int)

    parser.add_argument("--epoch_end", default=1000,
                    help="end at",
                    type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    # ONLY MODIFY SETTING HERE
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    batch_size = 1 # 128
    learning_rate = 1e-5       # 256/512 lr=4.5e-6 from 71 epochs
    img_size = 256
    switch_weight = 0.1 # self-reconstruction : a2b/b2a = 10 : 1
    
    save_path = '{}_{}_{}_pair'.format(args.dataset, args.ed, args.ne)    # model dir
    print(save_path)

    # load data
    train_data = dataset_pair(args.root_dir, 'train', img_size, img_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    f = os.path.join(os.getcwd(), save_path, 'settingc_latest.pt')
    config = OmegaConf.load('/kaggle/working/VQ-I2I/config_comb.yaml')
    config.model.target = 'taming_comb.models.vqgan.VQModelCrossGAN_ADAIN'
    config.model.base_learning_rate = learning_rate
    config.model.params.embed_dim = args.ed
    config.model.params.n_embed = args.ne
    config.model.z_channels = args.z_channel
    config.model.resolution = 256
    model = instantiate_from_config(config.model)
    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)

    # اگر بیش از یک GPU شناسایی شد، از DataParallel استفاده می‌کنیم
    if torch.cuda.device_count() > 1:
        print("Multiple GPUs detected, using DataParallel")
        model = torch.nn.DataParallel(model)  # اینجا از DataParallel استفاده می‌کنیم

    model.train()

    # print(model.loss.discriminator)
    
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+ 
                                list(model.decoder_a.parameters())+ 
                                list(model.decoder_b.parameters())+ 
                                list(model.quantize.parameters())+ 
                                list(model.quant_conv.parameters())+ 
                                list(model.post_quant_conv.parameters())+ 
                                list(model.style_enc_a.parameters())+ 
                                list(model.style_enc_b.parameters())+ 
                                list(model.mlp_a.parameters())+ 
                                list(model.mlp_b.parameters()),
                                lr=learning_rate, betas=(0.5, 0.999))
    
    opt_disc_a = torch.optim.Adam(model.loss_a.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.999))
    
    opt_disc_b = torch.optim.Adam(model.loss_b.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.999))

    if(os.path.isfile(f)):
        print('load ' + f)
        opt_ae.load_state_dict(ck['opt_ae_state_dict'])
        opt_disc_a.load_state_dict(ck['opt_disc_a_state_dict'])
        opt_disc_b.load_state_dict(ck['opt_disc_b_state_dict'])


    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)

    train_ae_a_error = []
    train_ae_b_error = []
    train_disc_a_error = []
    train_disc_b_error = []
    train_disc_a2b_error = []
    train_disc_b2a_error = []
    train_res_rec_error = []
    
    train_style_a_loss = []
    train_style_b_loss = []
    train_content_loss = []
    train_cross_recons_loss = []

    iterations = len(train_data) // batch_size
    iterations = iterations + 1 if len(train_data) % batch_size != 0 else iterations
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    for epoch in range(args.epoch_start, args.epoch_end+1):
        for i in range(iterations):

            dataA, dataB = next(iter(train_loader))
            dataA, dataB = dataA.to(device), dataB.to(device)

            ## Discriminator A
            opt_disc_a.zero_grad()
            
            s_a = model.encode_style(dataA, label=1)
            fakeA, _, _ = model(dataB, label=0, cross=True, s_given=s_a)

            recA, qlossA, _ = model(dataA, label=1, cross=False)
            
            b2a_loss, log = model.loss_a(_, dataA, fakeA, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")

            a_rec_d_loss, _ = model.loss_a(_, dataA, recA, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")
            
            disc_a_loss = 0.8*b2a_loss + 0.2*a_rec_d_loss
            disc_a_loss.backward()
            opt_disc_a.step()
            
            ## Discriminator B
            opt_disc_b.zero_grad()
            
            s_b = model.encode_style(dataB, label=0)
            fakeB, _, s_b_sampled = model(dataA, label=1, cross=True, s_given=s_b)

            recB, qlossB, _ = model(dataB, label=0, cross=False)
            
            a2b_loss, log = model.loss_b(_, dataB, fakeB, optimizer_idx=1, global_step=epoch,
                                    last_layer_
