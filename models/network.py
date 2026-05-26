import torch
import torch.nn as nn
from args import config

torch.autograd.set_detect_anomaly(True)

class HeightEstimationNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.captured = {}

        #====x,y Coordconv====
        use_coordconv = config.get('coordconv', False)
        if use_coordconv:
            self.add_coords = AddCoords3d()

        in_channels = 3 if use_coordconv else 1

        #====3d conv====
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
            # nn.Upsample(scale_factor=(2.0, 58/23, 58/16), mode='nearest'),
            nn.Upsample(size=(56, 58, 58), mode='nearest')
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.conv3d = nn.Conv3d(16, 1, 1)
        
        #====2D conv====
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


    def forward(self,x):
        if config['coordconv']:
            x = self.add_coords(x)

        x = self.encoder(x)
        x = self.decoder(x)
        x_cnn = self.conv3d(x)

        B, C, D, H, W = x_cnn.shape

        softmax_score = torch.softmax(x_cnn, dim=2) 
        
        x_softmax = softmax_score.detach()

        indices = torch.arange(D, device=x_cnn.device).float().view(1, 1, D, 1, 1)
        x_argmax = torch.sum(softmax_score * indices, dim=2)

        recon_output = self.refiner(x_argmax)
        
        if self.training:
            return recon_output
        else:
            return recon_output, x_cnn, x_argmax, x_softmax

    def apply(self, fn):
        self = super()._apply(fn)


class AddCoords3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        batch_size, _, depth, height, width = input_tensor.shape
        
        z_coords = torch.linspace(-1.0, 1.0, depth).type_as(input_tensor)
        y_coords = torch.linspace(-1.0, 1.0, height).type_as(input_tensor)
        x_coords = torch.linspace(-1.0, 1.0, width).type_as(input_tensor)

        _, y_mesh, x_mesh = torch.meshgrid(z_coords, y_coords, x_coords)

        y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        output_tensor = torch.cat([input_tensor, y_mesh, x_mesh], dim=1)
        
        return output_tensor


    