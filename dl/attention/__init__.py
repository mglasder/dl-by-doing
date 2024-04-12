import torch


def softmax(x: torch.Tensor, dim=None) -> torch.Tensor:
    powers = torch.e**x
    return powers / powers.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.tensor, key: torch.tensor, value: torch.tensor
) -> torch.tensor:
    E = query.shape[-1]
    qk = torch.matmul(query, torch.transpose(key, -1, -2))
    scaled_qk = qk / torch.sqrt(torch.tensor(E))
    scaled_softmax = softmax(scaled_qk, dim=-1)
    y = torch.matmul(scaled_softmax, value)
    return y
