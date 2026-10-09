import torch
import torch.nn as nn
import torch.nn.functional as F

class HeightEstimationNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.captured = {}

        self.use_coordconv = args.coordconv
        self.use_2d_cnn = args.use_2d_cnn

        # ==== x,y CoordConv ====
        if self.use_coordconv:
            self.add_coords = AddCoords3d()

        in_channels = 3 if self.use_coordconv else 1

        # ==== 3D conv ====
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Upsample(size=(56, 58, 58), mode='nearest')
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        )

        self.conv3d = nn.Conv3d(16, 1, 1)

        # ==== 2D conv ====
        if self.use_2d_cnn:
            self.refiner = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1)
            )

    def forward(self, x):
        if self.use_coordconv:
            x = self.add_coords(x)

        x = self.encoder(x)
        x = self.decoder(x)
        x_cnn = self.conv3d(x)

        B, C, D, H, W = x_cnn.shape

        softmax_score = torch.softmax(x_cnn, dim=2)

        x_softmax = softmax_score.detach()

        indices = torch.arange(D, device=x_cnn.device).float().view(1, 1, D, 1, 1)
        x_argmax = torch.sum(softmax_score * indices, dim=2)

        if self.use_2d_cnn:
            recon_output = self.refiner(x_argmax)
        else:
            recon_output = F.interpolate(x_argmax, scale_factor=2, mode='bilinear', align_corners=False)

        if self.training:
            return recon_output
        else:
            return recon_output, x_cnn, x_argmax, x_softmax

    def apply(self, fn):
        self = super()._apply(fn)



class AddCoords3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_y_coords", torch.empty(0), persistent=False)
        self.register_buffer("_x_coords", torch.empty(0), persistent=False)

    def forward(self, input_tensor):
        batch_size, _, depth, height, width = input_tensor.shape

        if (
            self._y_coords.shape != (1, 1, 1, height, 1)
            or self._x_coords.shape != (1, 1, 1, 1, width)
            or self._y_coords.device != input_tensor.device
            or self._x_coords.device != input_tensor.device
            or self._y_coords.dtype != input_tensor.dtype
            or self._x_coords.dtype != input_tensor.dtype
        ):
            self._y_coords = torch.linspace(
                -1.0, 1.0, height
            ).type_as(input_tensor).view(1, 1, 1, height, 1)

            self._x_coords = torch.linspace(
                -1.0, 1.0, width
            ).type_as(input_tensor).view(1, 1, 1, 1, width)

        y_mesh = self._y_coords.expand(
            batch_size, 1, depth, height, width
        )
        x_mesh = self._x_coords.expand(
            batch_size, 1, depth, height, width
        )

        return torch.cat([input_tensor, y_mesh, x_mesh], dim=1)