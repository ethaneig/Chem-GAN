import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphWGAN(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        discriminator_steps=1,
        generator_steps=2,
        gp_weight=10.0,
        batch_size=32
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.gp_weight = gp_weight
        self.latent_dim = generator.latent_dim
        self.batch_size = batch_size

        self.gen_loss_meter = []
        self.dis_loss_meter = []

    def compile(self, optimizer_generator, optimizer_discriminator):
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator

    def train_step(self, real_graphs):
        adjacency_real, features_real = real_graphs
        batch_size = adjacency_real.size(0)
        device = adjacency_real.device

        self.gen_loss_meter = []
        self.dis_loss_meter = []

        # Train discriminator
        for _ in range(self.discriminator_steps):
            z = torch.randn(batch_size, self.latent_dim, device=device)
            adj_fake, feat_fake = self.generator(z)

            logits_real = self.discriminator(adjacency_real, features_real)
            logits_fake = self.discriminator(adj_fake.detach(), feat_fake.detach())  # Detach for discriminator
            loss_d = torch.mean(logits_fake) - torch.mean(logits_real)

            gp = self.gradient_penalty(
                real_data=(adjacency_real, features_real),
                fake_data=(adj_fake.detach(), feat_fake.detach()),
                device=device
            )
            loss_d_total = loss_d + self.gp_weight * gp

            self.optimizer_discriminator.zero_grad()
            loss_d_total.backward()
            self.optimizer_discriminator.step()

            self.dis_loss_meter.append(loss_d_total.item())

        # Train generator
        for _ in range(self.generator_steps):
            z = torch.randn(batch_size, self.latent_dim, device=device)
            adj_fake, feat_fake = self.generator(z)
            
            logits_fake = self.discriminator(adj_fake, feat_fake)
            loss_g = -torch.mean(logits_fake)

            self.optimizer_generator.zero_grad()
            loss_g.backward()
            self.optimizer_generator.step()

            self.gen_loss_meter.append(loss_g.item())

        # Return metrics
        return {
            "loss_gen": sum(self.gen_loss_meter) / max(len(self.gen_loss_meter), 1),
            "loss_dis": sum(self.dis_loss_meter) / max(len(self.dis_loss_meter), 1),
        }

    def gradient_penalty(self, real_data, fake_data, device):
        adj_real, feat_real = real_data
        adj_fake, feat_fake = fake_data
        batch_size = adj_real.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha_feat = alpha.view(batch_size, 1, 1)
        
        adj_interp = adj_real * alpha + adj_fake * (1 - alpha)
        feat_interp = feat_real * alpha_feat + feat_fake * (1 - alpha_feat)
        
        adj_interp.requires_grad_(True)
        feat_interp.requires_grad_(True)
        
        disc_output = self.discriminator(adj_interp, feat_interp)
        
        grad_outputs = torch.ones_like(disc_output, device=device)
        gradients = torch.autograd.grad(
            outputs=disc_output,
            inputs=[adj_interp, feat_interp],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        
        grad_adj, grad_feat = gradients
        grad_adj_flat = grad_adj.view(batch_size, -1)
        grad_feat_flat = grad_feat.view(batch_size, -1)
        grad_all = torch.cat([grad_adj_flat, grad_feat_flat], dim=1)
        
        grad_norm = torch.sqrt(torch.sum(grad_all**2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((grad_norm - 1.0) ** 2)
        return gradient_penalty
