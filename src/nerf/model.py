import torch


class _BaseNeRFModel(torch.nn.Module):
    def __init__(self, Lx: int = 10, Ld: int = 4):
        """
        Initialize the NeRF model.
        """
        super().__init__()
        self.Lx = Lx
        self.Ld = Ld

        self.linear_layer_1 = torch.nn.Sequential(
            torch.nn.Linear(2*Lx*3, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
        )
        self.linear_layer_2 = torch.nn.Sequential(
            torch.nn.Linear(256 + 2*Lx*3, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
        )
        self.sigma_head = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.ReLU(inplace=True)
        )
        self.feature_head = torch.nn.Linear(256, 256)

        self.linear_layer_3 = torch.nn.Sequential(
            torch.nn.Linear(256 + 2*Ld*3, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid()
        )

    def positional_encoding(self, x: torch.Tensor, L: int) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor.
        
        Args:
            x: Input tensor of shape (..., 3).
            L: Number of frequency bands for encoding.

        Returns:
            Encoded tensor of shape (..., 2*3*L).
        """

        frequencies = 2.0 ** torch.arange(
            L, device=x.device, dtype=x.dtype
        )

        angles = torch.pi * x.unsqueeze(-1) * frequencies

        encoded = torch.stack(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1
        )

        return encoded.flatten(start_dim=-3)

    def forward(self, 
                points: torch.Tensor, 
                directions: torch.Tensor
        ):
        """
        Forward pass of the NeRF model.

        Args:
            points: 3D points/features.
            directions: View direction from the camera

        Returns:
            Features produced by the first NeRF trunk block.
        """
        # Add sin cos positional encoding to the input points.
        points_pos = self.positional_encoding(points, self.Lx)
        directions_pos = self.positional_encoding(directions, self.Ld)

        # Forward pass to get the density (sigma).
        features = self.linear_layer_1(points_pos)
        features = torch.cat([features, points_pos], dim=-1)
        features = self.linear_layer_2(features)
        sigma = self.sigma_head(features)
        features = self.feature_head(features)

        # Forward pass to get the color (RGB).
        features = torch.cat([features, directions_pos], dim=-1)
        rgb = self.linear_layer_3(features)

        return sigma, rgb


class FineNeRFModel(_BaseNeRFModel):
    def __init__(self, Lx: int = 10, Ld: int = 4):
        """
        Initialize the fine NeRF model.
        """
        super().__init__(Lx, Ld)


class CoarseNeRFModel(_BaseNeRFModel):
    def __init__(self, Lx: int = 10, Ld: int = 4):
        """
        Initialize the coarse NeRF model.
        """
        super().__init__(Lx, Ld)




        
        