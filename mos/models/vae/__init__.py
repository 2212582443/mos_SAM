from typing import List
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from itertools import islice


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class VanillaVAE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_chanels: int,
        latent_dim: int,
        shape=(128, 128),
        hidden_dims=[32, 64, 128, 256, 512],
        use_reparameterize=True,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.shape = shape
        self.use_reparameterize = use_reparameterize

        self.encoder = nn.Sequential(
            *[
                x
                for (in_ch, out_ch) in window([in_channels] + hidden_dims, 2)
                for x in [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(),
                ]
            ]
        )

        h, w = shape
        h //= 2 ** len(hidden_dims)
        w //= 2 ** len(hidden_dims)
        self.laten_shape = (hidden_dims[-1], h, w)

        self.fc_mu = nn.Linear(hidden_dims[-1] * h * w, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * h * w, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * h * w)
        hidden_dims.reverse()

        self.decoder = nn.Sequential(
            *[
                x
                for (in_ch, out_ch) in window(hidden_dims, 2)
                for x in [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(),
                ]
            ]
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_chanels, kernel_size=3, padding=1),
            nn.Conv2d(out_chanels, out_channels=out_chanels, kernel_size=3, padding=1),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder (bs,c,h,w)
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.laten_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # N(mu, sigma^2) ~ N(mu, var)
        # logvar = log(sigma^2) = 2 * log(sigma)
        #   log(sigma) = 0.5 * logvar
        #   sigma = exp(0.5 * logvar)
        sigma = torch.exp(0.5 * logvar)
        if self.use_reparameterize:
            eps = torch.randn_like(sigma)
            return eps * sigma + mu
        else:
            return sigma + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return [out, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss.detach(), "KLD": -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
