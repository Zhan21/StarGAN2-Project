import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import trange
from munch import Munch
from model import Discriminator, Generator, MappingNetwork, StyleEncoder

import warnings

warnings.simplefilter("ignore")


def test_lpips(lpips, loader, nets, args, device, test_iters=30):
    values = []

    for i in trange(test_iters):
        # fetch images and labels
        inputs = next(fetcher)
        x_real = inputs.x_src
        batch_size = x_real.shape[0]

        y_trg = torch.tensor(np.random.choice(np.arange(args.num_domains), size=batch_size))  # целевой домен
        y_org = torch.tensor(np.random.choice(np.arange(args.num_domains), size=batch_size))  # оригинальный домен

        x_real = x_real.to(device).float()
        y_trg, y_org = [x.to(device).long() for x in [y_trg, y_org]]

        # generate style -> generate image
        style_trg = nets.style_encoder(x_real, y_trg)
        x_fake = nets.generator(x_real, style_trg)

        values.append(lpips(x_fake.cpu(), x_real.cpu()).squeeze().item())
    return np.mean(values)


def build_optimizer(args, nets):
    optims = Munch()
    for net in nets.keys():
        optims[net] = torch.optim.Adam(
            params=nets[net].parameters(),
            lr=args.f_lr if net == "mapping_network" else args.lr,
            betas=[args.beta1, args.beta2],
            weight_decay=args.weight_decay,
        )
    return optims


def build_model(args, device):
    nets = Munch()
    # nets.generator = Generator(args.img_size).to(device)
    # nets.mapping_network = MappingNetwork(
    #     num_domains=args.num_domains).to(device)
    # nets.style_encoder = StyleEncoder(
    #     args.img_size, num_domains=args.num_domains).to(device)
    # nets.discriminator = Discriminator(
    #     args.img_size, num_domains=args.num_domains).to(device)

    nets.generator = nn.DataParallel(Generator(args.img_size).to(device))
    nets.mapping_network = nn.DataParallel(MappingNetwork(num_domains=args.num_domains).to(device))
    nets.style_encoder = nn.DataParallel(StyleEncoder(args.img_size, num_domains=args.num_domains).to(device))
    nets.discriminator = nn.DataParallel(Discriminator(args.img_size, num_domains=args.num_domains).to(device))
    return nets


def save_model(nets, iteration, save_path="./checkpoints"):
    torch.save(
        {
            "G": nets.generator.state_dict(),
            "D": nets.discriminator.state_dict(),
            "F": nets.mapping_network.state_dict(),
            "E": nets.style_encoder.state_dict(),
        },
        save_path + f"/{iteration}-weights.pth",
    )
    print(f"Saved model checkpoints into {save_path}...")


def load_model(args, device, save_path="./checkpoints"):
    nets = build_model(args, device)

    print(f"Loading the trained models from step {args.resume_iters}...")
    checkpoint_dict = torch.load(save_path + f"/{args.resume_iters}-weights.pth", map_location=torch.device(device))
    nets.generator.load_state_dict(checkpoint_dict["G"])
    nets.discriminator.load_state_dict(checkpoint_dict["D"])
    nets.mapping_network.load_state_dict(checkpoint_dict["F"])
    nets.style_encoder.load_state_dict(checkpoint_dict["E"])
    return nets


def generate_image(nets, x_real, y_trg, x_ref=None):
    with torch.no_grad():
        if x_ref:
            s_trg = nets.style_encoder(x_ref, y_trg)
        else:
            z_trg = torch.randn((args.batch_size, args.latent_dim)).to(device)
            s_trg = nets.mapping_network(z_trg, y_trg)

        x_fake = nets.generator(x_real, s_trg)

    return x_fake
