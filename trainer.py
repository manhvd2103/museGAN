import torch

from tqdm import tqdm
from typing import Iterable

from loss.criterion import compute_gradient_penalty

def train_one_step(generator, discriminator, g_optimizer, d_optimizer, real_samples, latent_dim, batch_size, device):
    latent = torch.randn(batch_size, latent_dim)
    real_samples = real_samples[0]
    fake_samples = generator(latent)

    # Train the discriminator network
    for _ in range(5):

        d_optimizer.zero_grad()

        # Get predictions from the discriminator
        prediction_real_d = discriminator(real_samples)
        prediction_fake_d = discriminator(fake_samples.detach())
        
        # Compute the losses for the discriminator network
        d_loss_real = -torch.mean(prediction_real_d)
        d_loss_fake = -torch.mean(prediction_fake_d)
        gradient_penalty = compute_gradient_penalty(discriminator, real_samples, fake_samples, device)
        d_loss = d_loss_fake + d_loss_real + 10 * gradient_penalty
        
        # Backpropagate the gradients
        d_loss.backward()

        # Update the weights
        d_optimizer.step()
    
    # Train the generator network
    g_optimizer.zero_grad()

    # Compute the losses for the generator network
    prediction_fake_g = discriminator(fake_samples)
    g_loss = - torch.mean(prediction_fake_g)

    # Backpropagate the gradients
    g_loss.backward()

    # Update the weights
    g_optimizer.step()

    return d_loss, g_loss

    


