"""Some implementation of VAE which was developed while reading and
    understanding the original paper by Kingma & Welling.
"""

import logging
from typing import Tuple, Optional
import torch
from torch import Tensor

logger = logging.getLogger(__name__)
logger = logging.getLogger("This")
logger.setLevel(logging.INFO)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0


class Encoder(torch.nn.Module):
    """Neural encoder class."""

    def __init__(self, img_size: Tuple[int, int]) -> None:
        super(Encoder, self).__init__()
        self.flatten_inp: torch.nn.Flatten = torch.nn.Flatten()
        self.inp_layer: torch.nn.Linear = torch.nn.Linear(in_features=img_size[0] * img_size[1], out_features=50)
        self.layer2_mu: torch.nn.Linear = torch.nn.Linear(in_features=50, out_features=25)
        self.layer2_sigma: torch.nn.Linear = torch.nn.Linear(in_features=50, out_features=25)
        self.distr: Optional[torch.distributions.Normal] = None
        self.tanh = torch.nn.Tanh()
        return None

    def __forward__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # encoder returns mu and log sigma^2 where sigma (both 1D vectors)
        x = self.flatten_inp(x)
        x = self.inp_layer(x)
        x = self.tanh(x)
        mu = self.layer2_mu(x)
        log_sigma_square = self.layer2_sigma(x)
        return (mu, log_sigma_square)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.__forward__(x)


class Decoder(torch.nn.Module):
    """Neural Decoder class."""

    def __init__(self) -> None:
        super(Decoder, self).__init__()
        # 25 -> 100 -> 784
        # no flatten layer needed
        self.inp_layer: torch.nn.Linear = torch.nn.Linear(in_features=25, out_features=100)
        self.layer_1_mu: torch.nn.Linear = torch.nn.Linear(in_features=100, out_features=784)
        self.layer_1_sigma: torch.nn.Linear = torch.nn.Linear(in_features=100, out_features=784)
        self.tanh = torch.nn.Tanh()
        return None

    def __forward__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.inp_layer(x)
        x = self.tanh(x)
        mu = self.layer_1_mu(x)
        log_sigma_square = self.layer_1_sigma(x)
        logger.info((mu.size(), log_sigma_square.size()))
        return (mu, log_sigma_square)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.__forward__(x)


class VAE(torch.nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec

    def __forward__(self, x: Tensor) -> Tensor:
        mu_enc: Tensor
        log_sigma_square_enc: Tensor
        mu_dec: Tensor
        log_sigma_square_dec: Tensor

        mu_enc, log_sigma_square_enc = self.enc(x)
        latent_distr: torch.distributions.MultivariateNormal = torch.distributions.MultivariateNormal(
            loc=mu_enc,
            covariance_matrix=torch.diag_embed(torch.exp(log_sigma_square_enc))
        )
        z: Tensor = latent_distr.rsample()
        mu_dec, log_sigma_square_dec = self.dec(z)
        reconstr_distr: torch.distributions.MultivariateNormal = torch.distributions.MultivariateNormal(
            loc=mu_dec,
            covariance_matrix=torch.diag_embed(torch.exp(log_sigma_square_dec))
        )
        reconstr_var: Tensor = reconstr_distr.rsample()
        return reconstr_var

    def __call__(self, x: Tensor):
        return self.__forward__(x)


def elbo(x_i, mu: Tensor, log_sigma_square: Tensor, log_prob: Tensor):
    # mu is mu parametrized by encoder
    # sigma is variance as parametrized by encoder
    # log prob is log p_{\theta}(x^{i} \mid z^{i}) or a Monte Carlo estimate based on this
    # where log p ... is just the pdf evald
    loss: Tensor = Tensor([0])
    ones: Tensor = torch.ones(log_sigma_square.size())
    mu_square: Tensor = mu.pow(2)
    intermed: Tensor = ones + log_sigma_square - mu_square - log_sigma_square.exp()
    intermed = intermed.sum(dim=1)
    final = torch.mean(- 1 / 2 * intermed + log_prob)

    print(f"Loss size: {loss.size()}")
    print(f"log_sigma_square size: {log_sigma_square.size()}")
    print(f"mu_square size: {mu_square.size()}")
    print(f"log_prob size: {log_prob.size()}")
    print(f"intermed size: {intermed.size()}")
    print(f"final size: {final.size()}")
    print(f"final: {final}")
    return final


def criterion(x: Tensor, x_hat: Tensor, mu: Tensor, log_sigma_square: Tensor):
    # rep_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    rep_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
    ones: Tensor = torch.ones(log_sigma_square.size())
    mu_square: Tensor = mu.pow(2)
    kl_div = -0.5 * torch.sum(ones + log_sigma_square - mu_square - log_sigma_square.exp())
    # return torch.mean((1 / 2) * torch.sum(ones + log_sigma_square - mu_square - log_sigma_square.exp())) + rep_loss
    return kl_div + rep_loss


def train(epochs: int, vae: VAE, dataloader: torch.utils.data.DataLoader):
    """Train vae for 'epochs' no. of epochs."""

    LEARNING_RATE = 1e-3
    optim: torch.optim.Adam = torch.optim.Adam(
        params=vae.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    vae.train()  # put to training mode (collect grads etc.)
    enc: Encoder = vae.enc
    dec: Decoder = vae.dec

    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(dataloader):
            optim.zero_grad()
            # posterior params
            enc_mu_sigma: Tuple[Tensor, Tensor] = enc(x)
            logger.info(f"@sample {batch_idx}: {enc_mu_sigma[0]}, {enc_mu_sigma[1]}")
            z: Tensor = torch.distributions.MultivariateNormal(
                loc=enc_mu_sigma[0],
                #covariance_matrix=torch.diag_embed(torch.exp(enc_mu_sigma[1]))).rsample(sample_shape=torch.Size([10]))  # I know this is ugly
                covariance_matrix=torch.diag_embed(torch.exp(enc_mu_sigma[1]))).rsample()  # I know this is ugly
            # reconstr distr params
            dec_mu_sigma: Tuple[Tensor, Tensor] = dec(z)
            rec_distr: torch.distributions.Distribution = torch.distributions.MultivariateNormal(
                loc=dec_mu_sigma[0], covariance_matrix=torch.diag_embed(torch.exp(dec_mu_sigma[1])))
            print(f"size of x: {x.size()}")
            x_hat = torch.sigmoid(rec_distr.rsample())
            x_hat = rec_distr.rsample()
            loss: Tensor = criterion(x.flatten(start_dim=1), x_hat, enc_mu_sigma[0], enc_mu_sigma[1])
            logger.info(f"Loss @ this stage: {loss}")
            print(f"Loss @ this stage: {loss}")
            loss.backward()
            optim.step()
        logger.info(f"Finished Iteration no {epoch}")
    return None
