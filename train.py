
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as torch_data

from dataset import REMADataset
from unet import UNet


def create_args():
    parser = argparse.ArgumentParser(
        description="train rema-su-resolution unet"
    )

    parser.add_argument(
        "--inp",
        type=str,
        help="input DEM.",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="out DEM with a different resolution. Scale factor 2.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate.",
        default=1e-4
    )

    parser.add_argument(
        "--lr-decay",
        type=float,
        help="lr decay factor.",
        default=1e-5
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        help="patch size training input.",
        default=256
    )

    parser.add_argument(
        "--scale-factor",
        type=int,
        help="scale factor of the output image.",
        default=2
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size.",
        default=32
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="epochs.",
        default=100
    )


    parser.add_argument(
        "--split",
        type=float,
        help="train test split (0, 1].",
        default=0.9
    )
    parser.add_argument(
        "--checkpoint-path",
        default="model",
        type=str,
        help="checkpoint path."
    )

    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        help="checkpoint prefix."
    )

    parser.add_argument(
        "--checkpoint-save-path",
        default="/home/kadh5719/development/git/rema_su_resolution/models/",
        type=str,
        help="path to save models in.",
    )


    return parser.parse_args()

def train(model, optimizer, device, data_loader):
    model.train()

    train_loss = 0.0
    tbar = tqdm(data_loader)
    num_samples = len(data_loader)

    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.to(device)

        output = model(image)
        loss = target - output
        loss = loss.sum()
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    return (train_loss/num_samples)

def test(model, device, data_loader):
    model.eval()
    val_loss = 0.0

    tbar = tqdm(data_loader)
    num_samples = len(data_loader)

    with torch.no_grad():
        for i, sample in enumerate(tbar):
                    
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = target - output
            loss = loss.sum()
            val_loss+=loss.item()
            tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))

    return (val_loss/num_samples)

if __name__=="__main__":
    args = create_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_dataset = REMADataset(
        input_pth=args.inp, target_pth=args.out,
        split=(1.0-args.split), scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(30, shear=30),
                    transforms.ToTensor()
                ])
    )

    test_dataset = REMADataset(
        input_pth=args.inp, target_pth=args.out,
        split=(1.0-args.split), scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(30, shear=30),
                    transforms.ToTensor()
                ])
    )

    train_dataloader = torch_data.DataLoader(
        train_dataset, num_workers=0, batch_size=args.batch_size, drop_last=True)

    test_dataloader = torch_data.DataLoader(
        test_dataset, num_workers=0, batch_size=args.batch_size, drop_last=True)


    model = UNet()
    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.Adam(
        lr=args.lr, weight_decay=args.lr_decay, params= model.parameters())

    train_log = []
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, device, train_dataloader)
        val_loss = test(model, device, test_dataloader)

        model_save_str = '{}/{}-{}.{}'.format(
            args.checkpoint_save_path, args.checkpoint_prefix, epoch, "pth")
        
        state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
        }

        torch.save(state, model_save_str)

        print(epoch, train_loss, val_loss)
        train_log.append([train_loss, val_loss])

        np.save("train_log_{}".format(args.checkpoint_prefix), train_log)

    print(epoch, train_loss, val_loss)
