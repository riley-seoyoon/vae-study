import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import functional as F
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import scglue.models as GLUE
import pandas as pd

# Define the RNAseq data
num_batch = 5
n_samples = 100
n_genes = 200
rna_data = torch.randn(n_genes, n_samples)

# Define the data loader
data = TensorDataset(rna_data)
data_loader = DataLoader(data, batch_size=num_batch, shuffle=True)
import torch
import torch.nn.functional as F
import torch.distributions as D
from abc import abstractmethod
from typing import Optional, Tuple

EPS = 1e-8


class DataEncoder(torch.nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        l = torch.sqrt(torch.tensor(x, dtype=torch.float))
        return l


    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x
    
    def forward(
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class DataDecoder(torch.nn.Module):
    def __init__(
            self, out_features: int, n_batches: int
    ) -> None:
        super().__init__()
        self.scale_lin = torch.nn.Parameter(torch.zeros(1, n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.lin_dec = torch.nn.Linear(out_features, out_features)

    def forward(
            self, u: torch.Tensor, 
            # v: torch.Tensor,
            b: torch.Tensor, 
            l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * self.lin_dec(u) + self.bias[b]
        # logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )



class Prior(torch.nn.Module):
    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)
# from sc import *

latent_feature = 10
num_epochs = 100
encoder = DataEncoder(in_features=n_samples, out_features=latent_feature)
decoder = DataDecoder(out_features=latent_feature, n_batches=num_batch)
# # Define the optimizer
lr = 0.1
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
prior = Prior()

def train_autoencoder(
    x: torch.Tensor,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_batch: int,
    prior: Optional[torch.nn.Module] = None,
    beta: float = 1.0,
    num_epochs: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Set the model to training mode
    encoder.train()
    decoder.train()

    # Create a data loader
    dataset = x
    loader = data_loader
    batch_size = n_genes // n_batch

    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()

            # Forward pass through the encoder and compute the latent distribution
            x_batch = batch[0]
            q_z_x, _ = encoder(x_batch, torch.tensor([]), lazy_normalizer=True)

            # Sample a latent variable z from the distribution
            z = q_z_x.rsample()

            # If a prior is given, compute the KL divergence between q(z|x) and p(z)
            if prior is not None:
                p_z = prior()
                kl_div = D.kl.kl_divergence(q_z_x, p_z).sum(dim=1)
                kl_loss = beta * kl_div.mean()
            else:
                kl_loss = 0.0

            # Decode the latent variable to get the reconstructed data distribution
            b = torch.zeros(batch_size, dtype=torch.long)

            p_x_z = decoder(z, b, l=torch.ones_like(x_batch))

            # Compute the negative log likelihood loss between the original data and the reconstructed data
            x_log_probs = p_x_z.log_prob(x_batch).sum(dim=1)
            nll_loss = -x_log_probs.mean()

            # Compute the total loss and backpropagate the gradients
            loss = nll_loss + kl_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/len(dataset):.6f}")

    # Set the model to evaluation mode
    encoder.eval()
    decoder.eval()

    # Encode the data to get the latent variables
    with torch.no_grad():
        q_z_x, _ = encoder(x, torch.tensor([]), lazy_normalizer=True)
        z = q_z_x.rsample()

        # Decode the latent variables to get the reconstructed data distribution
        b = torch.zeros(x.size(0), dtype=torch.long)
        p_x_z = decoder(z, b, l=torch.ones_like(x))

        # Compute the negative log likelihood loss between the original data and the reconstructed data
        x_log_probs = p_x_z.log_prob(x).sum(dim=1)
        nll_loss = -x_log_probs.mean()

    return z, p_x_z.mean, nll_loss

test_vae = train_autoencoder(data, encoder, decoder, optimizer, num_batch, prior=prior, num_epochs=100)