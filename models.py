"""VAE and Koopman-VAE for clock images (64x64 by default; 32x32 + half width for fast experiments)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


CHANNELS = [32, 64, 128, 256]


def _channels(size, width):
    """Conv channels for an image of `size` px (64 or 32), scaled by `width`; downsample to 4x4."""
    n = int(np.log2(size // 4))
    return [max(8, int(c * width)) for c in CHANNELS[:n]]


class Encoder(nn.Module):
    def __init__(self, d, size=64, width=1.0):
        super().__init__()
        ch = [1] + _channels(size, width)
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):              # size -> size/2 -> ... -> 4
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        h = max(64, int(256 * width))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(ch[-1] * 16, h), nn.SiLU(), nn.Linear(h, 2 * d))

    def forward(self, x):
        mu, logvar = self.fc(self.conv(x)).chunk(2, -1)
        return mu, logvar.clamp(-12, 6)


class Decoder(nn.Module):
    def __init__(self, d, size=64, width=1.0):
        super().__init__()
        ch = _channels(size, width)[::-1]               # e.g. 256, 128, 64, 32
        self.c0 = ch[0]
        h = max(64, int(256 * width))
        self.fc = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, ch[0] * 16), nn.SiLU())
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):              # 4 -> 8 -> ... -> size/2
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU()]
        layers += [nn.ConvTranspose2d(ch[-1], 1, 4, 2, 1)]  # -> size (logits)
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, self.c0, 4, 4))


class VAE(nn.Module):
    def __init__(self, d=4, size=64, width=1.0):
        super().__init__()
        self.d, self.size, self.width = d, size, width
        self.enc, self.dec = Encoder(d, size, width), Decoder(d, size, width)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.dec(z), mu, logvar


class KoopmanVAE(VAE):
    """A VAE whose latent space additionally evolves *linearly*: z_{t+1} = K z_t."""

    def __init__(self, d=4, size=64, width=1.0):
        super().__init__(d, size, width)
        self.K = nn.Parameter(torch.eye(d))

    def rollout(self, z0, steps):
        """Returns [z0, K z0, K^2 z0, ...] of shape (B, steps+1, d)."""
        zs = [z0]
        for _ in range(steps):
            zs.append(zs[-1] @ self.K.T)
        return torch.stack(zs, 1)


def recon_loss(logits, x):
    """Bernoulli NLL summed over pixels, averaged over the batch."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.shape[0]


def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / mu.shape[0]


def load_data(path="data/clocks.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def to_tensor(frames_u8, device):
    return torch.from_numpy(frames_u8).float().div_(255.0).unsqueeze(-3).to(device)


def downsample(frames_u8, size):
    """Average-pool uint8 frames (..., 64, 64) down to (..., size, size)."""
    if size == 64:
        return frames_u8
    f = 64 // size
    sh = frames_u8.shape[:-2]
    return frames_u8.reshape(*sh, size, f, size, f).astype(np.float32).mean((-3, -1)).round().astype(np.uint8)


def load_model(path, device="cpu"):
    """Load a VAE or Koopman VAE checkpoint, whatever its size / width."""
    ck = torch.load(path, map_location=device)
    cls = KoopmanVAE if "K" in ck["state"] else VAE
    m = cls(ck["d"], ck.get("size", 64), ck.get("width", 1.0))
    m.load_state_dict(ck["state"], strict=cls is KoopmanVAE)
    return m.to(device).eval(), ck
