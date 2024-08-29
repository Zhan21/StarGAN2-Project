from config import build_args
from dataloader import build_dataset, get_train_loader, InputFetcher

from utils import build_model, build_optimizer
from trainer import Trainer

import torch


def main():
    args = build_args()

    dataset, dataset_ref = build_dataset(args)

    loader = get_train_loader(dataset, args)
    loader_ref = get_train_loader(dataset_ref, args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders = InputFetcher(loader, loader_ref, args.latent_dim, device)

    nets = build_model(args, device)

    optims = build_optimizer(args, nets)

    trainer = Trainer(args, nets, optims, dataloaders, device)

    trainer.train()


if __name__ == "__main__":
    main()
