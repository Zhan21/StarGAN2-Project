import torch
import torch.nn.functional as F
from munch import Munch


def adversarial_loss(logits, target):
    assert target in [1, 0]
    # * 0.9 label-smoothing
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def gradient_penalty(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def generate_style(nets, z_trg, y_trg):
    return nets.mapping_network(z_trg, y_trg)


def extract_style(nets, x_ref, y_trg):
    return nets.style_encoder(x_ref, y_trg)


def discriminator_loss(nets, args, x_real, y_org, y_trg, s_trg):
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adversarial_loss(out, 1)
    loss_reg = gradient_penalty(out, x_real)

    # with fake images
    with torch.no_grad():
        x_fake = nets.generator(x_real, s_trg)

    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adversarial_loss(out, 0)

    loss = loss_real + loss_fake + (args.lambda_reg * loss_reg)
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def generator_loss(nets, args, x_real, y_org, y_trg, s_trg, s_trg2):
    # adversarial loss
    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adversarial_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    x_fake2 = nets.generator(x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + (args.lambda_sty * loss_sty) - \
        (args.lambda_ds * loss_ds) + (args.lambda_cyc * loss_cyc)

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())
