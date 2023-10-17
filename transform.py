import torch

def transform_milne(out):

    out_transformed = out.clone()
    
    out_transformed[..., 0] = 1e3 * out[..., 0]
    out_transformed[..., 1] = 1e3 * torch.sigmoid(out[..., 1])
    out_transformed[..., 2] = 1e3 * out[..., 2]
    out_transformed[..., 3] = 20.0 * torch.tanh(out[..., 3])
    out_transformed[..., 4] = 10.0**out[..., 4]
    out_transformed[..., 5] = 0.5 * torch.sigmoid(out[..., 5])
    out_transformed[..., 6] = 0.1 * torch.sigmoid(out[..., 6])
    out_transformed[..., 7] = torch.sigmoid(out[..., 7])
    out_transformed[..., 8] = torch.sigmoid(out[..., 8])

    return out_transformed

def transform_gaussian(out):

    out[..., 0] = 1.3*torch.sigmoid(1.0 + out[..., 0])
    out[..., 1] = torch.sigmoid(out[..., 1])
    out[..., 2] = out[..., 2]
    out[..., 3] = 0.5 * torch.sigmoid(out[..., 3])
    out[..., 4] = 0.3 * torch.sigmoid(out[..., 4])

    return out