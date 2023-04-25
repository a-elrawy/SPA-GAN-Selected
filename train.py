import os
import time

from torch import load
from torch.optim import Adam
import torch

from model import feature_matching_loss, generator_loss, cycle_consistency_loss, discriminator_loss, identity_loss
from utils import plot_example


class Trainer:
    """Trainer class for SPA-GAN."""

    def __init__(self, dataloader, generator_g, generator_f, discriminator_x, discriminator_y):
        """Initialize the trainer.
        Args:
            dataloader: DataLoader object
            generator_g: Generator G
            generator_f: Generator F
            discriminator_x: Discriminator X
            discriminator_y: Discriminator Y
        """
        self.dataloader = dataloader
        self.generator_g = generator_g
        self.generator_f = generator_f
        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y

        # Optimizers
        self.generator_g_optimizer = Adam(generator_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.generator_f_optimizer = Adam(generator_f.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_x_optimizer = Adam(discriminator_x.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_y_optimizer = Adam(discriminator_y.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_checkpoint(self):
        # Check Checkpoint
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Load the pretrained generator
        if os.path.exists(os.path.join(checkpoint_dir, 'generator_g.pth')):
            self.generator_g.load_state_dict(load(os.path.join(checkpoint_dir, 'generator_g.pth')))
            self.generator_f.load_state_dict(load(os.path.join(checkpoint_dir, 'generator_f.pth')))
            self.discriminator_x.load_state_dict(load(os.path.join(checkpoint_dir, 'discriminator_x.pth')))
            self.discriminator_y.load_state_dict(load(os.path.join(checkpoint_dir, 'discriminator_y.pth')))

    def save_checkpoint(self):
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(self.generator_g.state_dict(), os.path.join(checkpoint_dir, 'generator_g.pth'))
        torch.save(self.generator_f.state_dict(), os.path.join(checkpoint_dir, 'generator_f.pth'))
        torch.save(self.discriminator_x.state_dict(), os.path.join(checkpoint_dir, 'discriminator_x.pth'))
        torch.save(self.discriminator_y.state_dict(), os.path.join(checkpoint_dir, 'discriminator_y.pth'))

    def train_step(self, real_x, real_y):
        """Train the model for one step.
        Args:
            real_x: Real image from domain X
            real_y: Real image from domain Y
        """
        orig_x = real_x.to(self.device)
        orig_y = real_y.to(self.device)

        # Reset gradients
        self.generator_g_optimizer.zero_grad()
        self.generator_f_optimizer.zero_grad()
        self.discriminator_x_optimizer.zero_grad()
        self.discriminator_y_optimizer.zero_grad()

        # Generate samples

        # Get feature map
        pred_real_x, feature_map_x = self.discriminator_x(orig_x, return_feature_map=True)
        pred_real_y, feature_map_y = self.discriminator_y(orig_y, return_feature_map=True)

        # Apply feature map to real image
        real_x = orig_x * feature_map_x
        real_y = orig_y * feature_map_y

        # Generate fake images
        fake_y, feat_x = self.generator_g(real_x, return_feat=True)
        rec_x, feat_y_hat = self.generator_f(fake_y, return_feat=True)

        plot_example(orig_y, fake_y, title1='real_y', title2='generated y')

        fake_x, feat_y = self.generator_f(real_y, return_feat=True)
        rec_y, feat_x_hat = self.generator_g(fake_x, return_feat=True)

        plot_example(orig_x, fake_x, title1='real_x', title2='generated x')
        # Identity mapping
        same_x = self.generator_f(real_x)
        same_y = self.generator_g(real_y)

        # Discriminator outputs
        pred_fake_x = self.discriminator_x(fake_x)
        pred_fake_y = self.discriminator_y(fake_y)

        # Calculate Loss

        # Feature Map Loss
        feat_x_loss = feature_matching_loss(feat_x, feat_x_hat)
        feat_y_loss = feature_matching_loss(feat_y, feat_y_hat)
        total_feat_loss = feat_x_loss + feat_y_loss

        # GAN loss
        g_loss_gan = generator_loss(pred_fake_y)
        f_loss_gan = generator_loss(pred_fake_x)

        # Cycle loss
        cycle_loss = cycle_consistency_loss(orig_x, rec_x) + cycle_consistency_loss(orig_y, rec_y)

        # total loss
        total_g_loss = g_loss_gan + cycle_loss + identity_loss(orig_y, same_y) + total_feat_loss
        total_f_loss = f_loss_gan + cycle_loss + identity_loss(orig_x, same_x) + total_feat_loss

        discriminator_x_loss = discriminator_loss(pred_real_x, pred_fake_x)
        discriminator_y_loss = discriminator_loss(pred_real_y, pred_fake_y)

        # Perform backprop
        total_g_loss.backward(retain_graph=True)
        total_f_loss.backward(retain_graph=True)

        discriminator_x_loss.backward(retain_graph=True)
        discriminator_y_loss.backward(retain_graph=True)

        # Update weights
        self.generator_g_optimizer.step()
        self.generator_f_optimizer.step()

        self.discriminator_x_optimizer.step()
        self.discriminator_y_optimizer.step()

        return total_g_loss.item(), total_f_loss.item(), discriminator_x_loss.item(), discriminator_y_loss.item()

    def train(self, epochs=10):
        """Train the model.
        Args:
            epochs: Number of epochs to train for
        """

        # To CUDA
        self.generator_g.to(self.device)
        self.generator_f.to(self.device)
        self.discriminator_x.to(self.device)
        self.discriminator_y.to(self.device)

        self.load_checkpoint()
        # Start training
        for epoch in range(epochs):
            start = time.time()
            for i, (image_x, image_y) in enumerate(self.dataloader):
                g_loss, f_loss, d_x_loss, d_y_loss = self.train_step(image_x, image_y)
                if i % 10 == 0:
                    print(
                        f'Epoch: {epoch}/{epochs}, Step: {i}/{len(self.dataloader)}, g_loss: {g_loss}, f_loss: {f_loss},'
                        f'd_x_loss: {d_x_loss}, d_y_loss: {d_y_loss}'
                    )
                    self.save_checkpoint()

            print(f'Time taken for epoch {epoch + 1} is {time.time() - start} sec\n')

            # Save the model checkpoints

    def generate(self, image, generator='g'):
        """Generate images from the given image.
        Args:
            image: Image to generate from
            generator: Type of generator to use
        Returns:
            Generated image
        """
        image = image.to(self.device)
        return self.generator_g(image) if generator == 'g' else self.generator_f(image)
