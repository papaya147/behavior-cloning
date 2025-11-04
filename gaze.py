import torch

def gaussian_mask(points, height, width, sigma=None, normalize=True):
    if sigma is None:
        sigma = min(height, width) / 6.0

    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)

    batched = points.dim() == 2
    if not batched:
        points = points.unsqueeze(0)

    batch_size = points.shape[0]
    
    center_x = points[:, 0] * width
    center_y = points[:, 1] * height

    center_x = center_x.view(batch_size, 1, 1)
    center_y = center_y.view(batch_size, 1, 1)

    y = torch.arange(height).view(-1, 1).unsqueeze(0)
    x = torch.arange(width).view(1, -1).unsqueeze(0)

    d2 = (x - center_x) ** 2 + (y - center_y) ** 2

    mask = torch.exp(-d2 / (2 * sigma ** 2))

    if normalize:
        max_vals = mask.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
        mask = mask / max_vals

    if not batched:
        mask = mask.squeeze(0)

    return mask