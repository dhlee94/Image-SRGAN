from utils.utils import seed_everything, save_checkpoint
import argparse
import pickle
import numpy as np
from core.optimizer import CosineAnnealingWarmUpRestarts
from core.function import train, test
from data.dataset import ImageDataset
from timm.models.layers import to_2tuple
from torch.utils.data import DataLoader
from models.Generator import Generator
from models.Discriminator import SimpleDiscriminator
from models.Extract_feature_map import FeatureExtractor
import torch
import torch.nn as nn
import os
import albumentations
import albumentations.pytorch
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy
import pandas as pd
import cv2
import sys
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    default=304, help='random seed')
parser.add_argument('--csv_path', type=str, required=True, metavar="FILE", help='path to CSV file')
parser.add_argument('--world_size', type=int, default=1, help='Number of Process for multi-GPU')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')
parser.add_argument('--log-path', default='./log', type=str, help='Write Log Path')
parser.add_argument('--num_class', type=int, default=1, help='Number of Class')
parser.add_argument('--batch_size', type=int, default=1, help='Number of Batch Size')
parser.add_argument('--epoch', default=100, type=int, help='Number of Epoch')
parser.add_argument('--workers', type=int, default=1, help='Number of Workers')
parser.add_argument('--input_channel', type=int, default=3, help='Number of Input Channel')
parser.add_argument('--optim', default='SGD', type=str, help='type of optimizer')
parser.add_argument('--momentum', default=0.95, type=float, help='SGD momentum')
parser.add_argument('--lr', default=1e-4, type=float, help='Train Learning Rate')
parser.add_argument('--optimizer_eps', default=1e-8, type=float, help='AdamW optimizer eps')
parser.add_argument('--optimizer_betas', default=(0.9, 0.999), help='AdamW optimizer betas')
parser.add_argument('--weight_decay', default=0.95, type=float, help='AdamW optimizer weight decay')
parser.add_argument('--scheduler', default='LambdaLR', type=str, help='type of Scheduler')
parser.add_argument('--lambda_weight', default=0.975, type=float, help='LambdaLR Scheduler lambda weight')
parser.add_argument('--t_scheduler', default=80, type=int, help='CosineAnnealingWarmUpRestarts optimizer time step')
parser.add_argument('--trigger_scheduler', default=1, type=int, help='CosineAnnealingWarmUpRestarts optimizer T trigger')
parser.add_argument('--eta_scheduler', default=1.25e-3, type=float, help='CosineAnnealingWarmUpRestarts optimizer eta max')
parser.add_argument('--up_scheduler', default=8, type=int, help='CosineAnnealingWarmUpRestarts optimizer time Up')
parser.add_argument('--gamma_scheduler', default=0.5, type=float, help='CosineAnnealingWarmUpRestarts optimizer gamma')
parser.add_argument('--model_path', default='./weight/best_model.pth', type=str, help='Model Path')
parser.add_argument('--model_save_path', default='./weight', type=str, help='Model Save Path')
parser.add_argument('--retrain', default=False, type=bool, help='Model Save Path')
parser.add_argument('--write_iter_num', default=10, type=int, help='Write iter num')
parser.add_argument('--lambda_pixel', default=1, type=int, help='lambda pixel Weight')
parser.add_argument('--lambda_adv', default=1, type=int, help='lambda adv Weihgt')
parser.add_argument('--warmup_batches', default=1000, type=int, help='warmup batches params')

def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    args.gpu = gpu
    log_path = args.log_path
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_csv_path = pd.read_csv(os.path.join(args.csv_path, 'Train_data.csv'))
    
    image_shape = to_2tuple(args.img_size)
    height, width = image_shape

    train_transform = albumentations.Compose(
            [
                albumentations.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.ShiftScaleRotate(p=1, rotate_limit=90),
                    albumentations.VerticalFlip(p=1),
                    albumentations.RandomBrightnessContrast(p=1),
                    albumentations.GaussNoise(p=1)                    
                ],p=1)
            ],
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            albumentations.pytorch.ToTensorV2()            
        )
    generator = Generator(img_size=args.img_size, channels=args.input_channel, out_channels=input_channel, 
                          filtersize=[64, 128, 256, 512], up_mode="bilinear", check_sigmoid=False)
    discriminator = SimpleDiscriminator()
    feature_extrator = FeatureExtractor()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=args.optimizer_betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=args.optimizer_betas)

    Gscheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer_G, lr_lambda=lambda epoch: args.lambda_weight ** epoch)
    Dscheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer_D, lr_lambda=lambda epoch: args.lambda_weight ** epoch)

    train_dataset = ImageDataset(Data_path=train_csv_path, transform=train_transform)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=True)
    


    start_epoch = 0
    best_acc = 0
    if args.retrain:
        checkpoint = torch.load(args.model_path, map_location={'cuda:0':'cpu'})
        start_epoch = checkpoint['epoch']
        discriminator.load_state_dict(checkpoint['Dstate_dict'])
        Doptimizer.load_state_dict(checkpoint['Doptimizer'])
        Dscheduler.load_state_dict(checkpoint['Dscheduler'])
        generator.load_state_dict(checkpoint['Gstate_dict'])
        Goptimizer.load_state_dict(checkpoint['Goptimizer'])
        Gscheduler.load_state_dict(checkpoint['Gscheduler'])
        
    for epoch in range(start_epoch, args.epoch):
        file = open(os.path.join(log_path, f'{epoch}_log.txt'), 'a')
        train(generator=generator, discriminator=discriminator, feature_extrator=feature_extrator, write_iter_num=args.write_iter_num, device=device,
              train_dataset=trainloader, optimizer_G=optimizer_G, optimizer_D=optimizer_D, criterison_G=nn.L1Loss(), criterison_C=nn.L1Loss(), criterison_P=nn.L1Loss(), 
              lambda_pixel=args.lambda_pixel, lambda_adv=args.lambda_adv, warmup_batches=args.warmup_batches, file=file)

        Gscheduler.step()
        Dscheduler.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'Gstate_dict': generator.state_dict(),
            'Dstate_dict': discriminator.state_dict(),
            'Goptimizer' : Goptimizer.state_dict(),
            'Gscheduler' : Gscheduler.state_dict(),
            'Doptimizer' : Doptimizer.state_dict(),
            'Dscheduler' : Dscheduler.state_dict()
        }, is_best=False, path=args.model_save_path)
        file.close()

if __name__ == '__main__':
    main()