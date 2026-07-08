import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    # two spectral-normalized convs, used inside ProxNet
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        )
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class ProxNet(nn.Module):
    # learned proximal operator: ProxNet(Q) = Q - R(Q)
    # spectral norm keeps R non-expansive
    def __init__(self, hidden_channels: int = 16, num_blocks: int = 2):
        super().__init__()
        self.lift = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=True)
        )
        self.act = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.project = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, q):
        feat = self.act(self.lift(q))
        for block in self.blocks:
            feat = feat + block(feat)
        residual = self.project(feat)
        return q - residual


class FDUCore(nn.Module):
    # FISTA unfolding on a single real channel
    # optimization variable is the Cartesian pair (X, Y), never amplitude/phase
    def __init__(
        self,
        num_stages: int = 8,
        proxnet_hidden: int = 16,
        proxnet_blocks: int = 2,
        gamma: float = 2.2,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.gamma = gamma

        # per-stage ProxNet pair: theta_k^X, theta_k^Y
        self.proxnet_x = nn.ModuleList(
            [
                ProxNet(hidden_channels=proxnet_hidden, num_blocks=proxnet_blocks)
                for _ in range(num_stages)
            ]
        )
        self.proxnet_y = nn.ModuleList(
            [
                ProxNet(hidden_channels=proxnet_hidden, num_blocks=proxnet_blocks)
                for _ in range(num_stages)
            ]
        )

        # step-size logits psi_k; eta_k = sigmoid(psi_k)/2 in (0, 1/2)
        self.psi = nn.Parameter(torch.zeros(num_stages))

    @staticmethod
    def _fft_real_imag(img):
        spec = torch.fft.fft2(img, norm="ortho")
        return spec.real, spec.imag

    @staticmethod
    def _ifft_from_real_imag(x, y):
        spec = torch.complex(x, y)
        return torch.fft.ifft2(spec, norm="ortho").real

    def forward(self, low_light_img, return_intermediate: bool = False):
        b, c, h, w = low_light_img.shape
        assert c == 1, "FDUCore operates on a single real-valued channel."

        x_prev2, y_prev2 = self._fft_real_imag(low_light_img)

        init_img = low_light_img.clamp(min=1e-6).pow(1.0 / self.gamma)
        x0, y0 = self._fft_real_imag(init_img)

        s_prev = torch.tensor(1.0, device=low_light_img.device)
        intermediates = [] if return_intermediate else None

        x_cur, y_cur = x_prev2, y_prev2
        for k in range(self.num_stages):
            # momentum extrapolation
            s_next = (1.0 + torch.sqrt(1.0 + 4.0 * s_prev**2)) / 2.0
            coeff = (s_prev - 1.0) / s_next
            p_x = x_cur + coeff * (x_cur - x_prev2)
            p_y = y_cur + coeff * (y_cur - y_prev2)

            # gradient descent, closed form
            eta_k = torch.sigmoid(self.psi[k]) / 2.0
            q_x = (1.0 - 2.0 * eta_k) * p_x + 2.0 * eta_k * x0
            q_y = (1.0 - 2.0 * eta_k) * p_y + 2.0 * eta_k * y0

            # learned proximal step
            x_next = self.proxnet_x[k](q_x)
            y_next = self.proxnet_y[k](q_y)

            if return_intermediate:
                intermediates.append(self._ifft_from_real_imag(x_next, y_next))

            x_prev2, y_prev2 = x_cur, y_cur
            x_cur, y_cur = x_next, y_next
            s_prev = s_next

        enhanced = self._ifft_from_real_imag(x_cur, y_cur)
        if return_intermediate:
            return enhanced, intermediates
        return enhanced


class Model(nn.Module):
    # FDU-Net applied independently to each RGB channel
    def __init__(
        self,
        num_stages: int = 8,
        proxnet_hidden: int = 16,
        proxnet_blocks: int = 2,
        gamma: float = 2.2,
        share_channel_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.share_channel_weights = share_channel_weights

        def make_core():
            return FDUCore(
                num_stages=num_stages,
                proxnet_hidden=proxnet_hidden,
                proxnet_blocks=proxnet_blocks,
                gamma=gamma,
            )

        if share_channel_weights:
            shared = make_core()
            self.cores = nn.ModuleList([shared, shared, shared])
        else:
            self.cores = nn.ModuleList([make_core() for _ in range(3)])

    def forward(self, x, return_intermediate: bool = False):
        b, c, h, w = x.shape
        assert c == 3, f"NoirNetASP expects 3-channel RGB input, got {c} channels."

        channel_outputs = []
        channel_intermediates = [] if return_intermediate else None

        for ch in range(3):
            channel_img = x[:, ch : ch + 1, :, :]
            if return_intermediate:
                out_ch, inter_ch = self.cores[ch](channel_img, return_intermediate=True)
                channel_intermediates.append(inter_ch)
            else:
                out_ch = self.cores[ch](channel_img, return_intermediate=False)
            channel_outputs.append(out_ch)

        enhanced = torch.cat(channel_outputs, dim=1)

        if not return_intermediate:
            return enhanced

        # reassemble per-stage RGB images from per-channel lists
        intermediates = []
        for k in range(self.num_stages):
            stage_rgb = torch.cat(
                [channel_intermediates[ch][k] for ch in range(3)], dim=1
            )
            intermediates.append(stage_rgb)

        return enhanced, intermediates

    @torch.no_grad()
    def get_fourier_targets(self, target_rgb):
        # ground-truth (X, Y) per channel, for Fourier-domain supervision
        b, c, h, w = target_rgb.shape
        assert c == 3
        x_list, y_list = [], []
        for ch in range(3):
            spec = torch.fft.fft2(target_rgb[:, ch : ch + 1, :, :], norm="ortho")
            x_list.append(spec.real)
            y_list.append(spec.imag)
        return torch.cat(x_list, dim=1), torch.cat(y_list, dim=1)

    def get_fourier_prediction(self, enhanced_rgb):
        x_list, y_list = [], []
        for ch in range(3):
            spec = torch.fft.fft2(enhanced_rgb[:, ch : ch + 1, :, :], norm="ortho")
            x_list.append(spec.real)
            y_list.append(spec.imag)
        return torch.cat(x_list, dim=1), torch.cat(y_list, dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_stages=4, proxnet_hidden=8, proxnet_blocks=1).to(device)

    b, h, w = 2, 64, 64
    low = torch.rand(b, 3, h, w, device=device)
    target = torch.rand(b, 3, h, w, device=device)

    enhanced, intermediates = model(low, return_intermediate=True)
    assert enhanced.shape == (b, 3, h, w)
    assert len(intermediates) == 4

    for core in model.cores:
        eta = torch.sigmoid(core.psi) / 2.0
        assert torch.all(eta > 0) and torch.all(eta < 0.5)

    loss = F.l1_loss(enhanced, target)
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
