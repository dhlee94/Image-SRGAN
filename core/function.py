import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import os
import time
import wandb
import pickle
import logging
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.utils import AverageMeter, batch_accuracy
from .criterion import *
from .optimizer import *
import copy

def train(generator=None, discriminator=None, feature_extrator=None, write_iter_num=10, device=torch.device('cpu'), train_dataset=None, 
          optimizer_G=None, optimizer_D=None, criterison_G=nn.L1Loss(), criterison_C=nn.L1Loss(), criterison_P=nn.L1Loss(), 
          lambda_pixel=1, lambda_adv=1, warmup_batches=1000, file=None):
    scaler = torch.cuda.amp.GradScaler()
    assert train_dataset is not None, print("train_dataset is none")
    ave_accuracy = AverageMeter()
    #scaler = torch.cuda.amp.GradScaler()
    batches_done = 0
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    feature_extrator = feature_extrator.to(device)
    generator.train()
    discriminator.train()
    for idx, (Image, Label) in enumerate(tqdm(train_dataset)):
        #model input data
        img_hr, img_lr = train_batch
        img_hr = img_hr.to(device, non_blocking=True)
        img_lr = img_lr.to(device, non_blocking=True)

        #Adversarial ground truths
        valid = Variable(Tensor(np.ones((img_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((img_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        #Generator
        
        gen_hr = generator(img_lr)
        optimizer_G.zero_grad()
        loss_pixel = criterison_P(gen_hr, img_hr)
        accuracy = batch_accuracy(gen_hr, img_hr)
        if batches_done < warmup_batches:
            loss_pixel.backward()
            optimizer_G.step()
            if idx % write_iter_num == 0:
                tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(train_dataset)} '
                        f'Generator Loss : {loss_pixel :.4f} ')
            batches_done += 1
            continue

        pred_real = discriminator(img_hr).detach()
        pred_fake = discriminator(gen_hr)
        loss_GAN = criterison_G(pred_fake - pred_real.mean(0, keepdim=True), valid)

        gen_features = feature_extrator(gen_hr)
        real_features = feature_extrator(img_hr)
        loss_content = criterison_C(gen_features, real_features.detach())

        loss_T = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel
        loss_T.backward()
        optimizer_G.step()

        #Discriminator
        optimizer_D.zero_grad()
        gen_hr = gen_hr.detach().requires_grad_(False).contiguous()
        pred_real = discriminator(img_hr)
        pred_fake = discriminator(gen_hr)
        loss_real = criterison_G(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterison_G(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = 0.5*loss_real + 0.5*loss_fake
        loss_D.backward()
        optimizer_D.step()
        
        if idx % write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(train_dataset)} '
                        f'Generator Loss : {loss_T :.4f} ' f'Discriminator Loss : {loss_D :.4f} '
                        f'Generator content Loss : {loss_content :4f} ' f'Generator pixel Loss : {loss_pixel :4f} '
                        f'Generator BCE Loss : {loss_GAN :4f} '
                        f'Generator Best Acc : {accuracy:4f}')
        if idx % 2*write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(train_dataset)} '
                        f'Generator Loss : {loss_T :.4f} ' f'Discriminator Loss : {loss_D :.4f} '
                        f'Generator content Loss : {loss_content :4f} ' f'Generator pixel Loss : {loss_pixel :4f} '
                        f'Generator BCE Loss : {loss_GAN :4f} '
                        f'Generator Best Acc : {accuracy:4f}', file=file)