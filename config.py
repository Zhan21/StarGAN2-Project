from munch import Munch


def build_args():
    args = Munch()

    # Dataset
    args.img_size = 256
    args.num_workers = 4

    # Models
    args.lambda_reg = 1
    args.lambda_cyc = 1
    args.lambda_sty = 1
    args.lambda_ds = 1
    args.latent_dim = 16
    args.num_domains = 10

    # Optimizers
    args.lr = 1e-4
    args.f_lr = 1e-6
    args.beta1 = 0.0
    args.beta2 = 0.99
    args.weight_decay = 1e-4

    # Trainer
    args.total_iters = 100000
    args.resume_iter = 0
    args.batch_size = 8
    args.val_batch_size = 32
    args.weight_decay = 1e-4
    args.num_outs_per_domain = 10
    args.ds_iter = 100000

    # Logging
    args.print_every = 10
    args.sample_every = 5000
    args.save_every = 10000
    args.eval_every = 50000
    args.lpips_every = 100

    return args
