import wandb
import time
import datetime
from tqdm.auto import trange
from lpips_pytorch import LPIPS

from utils import load_model, save_model, test_lpips
from loss import discriminator_loss, generator_loss, generate_style, extract_style


class Trainer:
    def __init__(
        self,
        args,
        nets,
        optims,
        dataloader,
        device
    ):
        super().__init__()
        self.args = args
        self.nets = nets
        self.optims = optims
        self.dataloader = dataloader
        self.device = device

        self.initial_lambda_ds = self.args.lambda_ds
        self.lpips = LPIPS()

        wandb.login()
        run = wandb.init(project="StarGANv2")

        self.wandb = wandb

    def train(self):
        if self.args.resume_iter > 0:
            load_model(self.args, self.device)

        start_time = time.time()
        for i in trange(self.args.resume_iter, self.args.total_iters):
            inputs = next(self.dataloader)

            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # _____________________Discriminator_____________________

            # 1) Generate style F(z_trg) -> style_trg
            s_trg = generate_style(nets, z_trg, y_trg)

            # Compute losses:
            # generate image G(x_real, style_trg) -> x_fake
            # discriminate D(x_fake, y_origin) -> loss_fake
            # discriminate D(x_real, y_origin) -> loss_real

            d_loss, d_losses_latent = discriminator_loss(
                self.nets, self.args, x_real, y_org, y_trg, s_trg)

            self._zero_grads()
            d_loss.backward()
            self.optims.discriminator.step()

            # 2) Extract style E(x_ref, y_target) -> style_trg
            s_trg = extract_style(nets, x_ref, y_trg)

            # Compute losses
            d_loss, d_losses_ref = discriminator_loss(
                self.nets, self.args, x_real, y_org, y_trg, s_trg)

            self._zero_grads()
            d_loss.backward()
            self.optims.discriminator.step()

            # _______________________Generator_______________________

            # 3) Generate style from noise using Mapping Network
            s_trg = generate_style(nets, z_trg, y_trg)
            s_trg2 = generate_style(nets, z_trg2, y_trg)

            # Compute losses:
            # generate style1 -> generate fake_image1 with injected style1
            # extract style from fake_image1 -> rec_style
            # > rec_loss(style1 - rec_style)

            # generate style2 -> generate fake_image2 with injected style2
            # > diversity_loss(fake_image1 - fake_image2.detach())

            # extract style_origin from real_image and y_origin domain
            # generate rec_image from fake_image1 with injected style_origin
            # > cycle_loss(rec_image - real_image)

            g_loss, g_losses_latent = generator_loss(
                self.nets, self.args, x_real, y_org, y_trg, s_trg, s_trg2)

            self._zero_grads()
            g_loss.backward()
            self.optims.generator.step()
            self.optims.mapping_network.step()
            self.optims.style_encoder.step()

            # 4) Extract style from reference image using Style Encoder
            s_trg = extract_style(nets, x_ref, y_trg)
            s_trg2 = extract_style(nets, x_ref2, y_trg)

            # Compute losses:
            g_loss, g_losses_ref = generator_loss(
                self.nets, self.args, x_real, y_org, y_trg, s_trg, s_trg2)

            self._zero_grads()
            g_loss.backward()
            self.optims.generator.step()

            # _________________Decay lambda diversity________________
            if self.args.lambda_ds > 0:
                self.args.lambda_ds -= (self.initial_lambda_ds /
                                        self.args.ds_iter)

            # ________________________Logging_________________________
            if (i+1) % self.args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (
                    elapsed, i+1, self.args.total_iters)

                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = self.args.lambda_ds
                display_losses = ['G/latent_adv', 'G/ref_cyc', 'G/ref_sty']
                log += ' '.join(['%s: [%.4f]' % (key, all_losses[key])
                                for key in display_losses])
                print(log)

                self.wandb.log(all_losses)

            # __________________Test LPIPS____________________
            if (i+1) % self.args.lpips_every == 0:
                lpips_value = test_lpips(self.lpips, self.dataloader, self.nets,
                                         self.args, self.device, test_iters=30)
                self.wandb.log({"test_lpips": lpips_value})

            # __________________Save Model____________________
            if (i+1) % self.args.save_every == 0:
                save_model(self.nets, i+1)

    def _zero_grads(self):
        for optim in self.optims.values():
            optim.zero_grad()
