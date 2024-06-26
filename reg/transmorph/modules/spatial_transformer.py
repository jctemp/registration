import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformerSeries(nn.Module):
    """
    Spatial Transformer for series of images
    """

    def __init__(self, size, mode="bilinear"):
        """
        Args:
            size(tuple[int]): size of the input images
            mode(str): interpolation mode
        """
        super().__init__()

        self.mode = mode

        grid_size = size[:-1]
        dvf_size = size[-1]

        # create sampling grid

        vectors = [torch.arange(0, s) for s in grid_size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)[..., None]
        grid = grid * torch.ones(dvf_size)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        """
        Args:
            src: original image series
            flow: 2d series displacment vector field
        """
        # new locations
        new_locs = self.grid + flow

        # exclude time dimension
        shape = flow.shape[2:-1]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # (B, C, W, H, D) -> (B, W, H, C, D)
        new_locs = new_locs.permute(0, 2, 3, 1, 4)
        new_locs = new_locs[..., [1, 0], :]

        # iterate over time dimension and resample each image
        resampled = torch.stack(
            [
                nnf.grid_sample(
                    src[:, :, :, :, i],
                    new_locs[:, :, :, :, i],
                    align_corners=True,
                    mode=self.mode,
                )
                for i in range(src.shape[-1])
            ]
        ).permute(1, 2, 3, 4, 0)

        return resampled


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        """
        Args:
            size(tuple[int]): size of the input images
            mode(str): interpolation mode
        """
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        """
        Args:
            src: original image series
            flow: n-d displacment vector field
        """
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
