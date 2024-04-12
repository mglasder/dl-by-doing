import pytest
import torch
from dl.attention import scaled_dot_product_attention


@pytest.mark.parametrize(
    "size_q, size_k, size_v",
    [
        # (N, ..., L, E); (N, ..., S, E); (N, ..., S, Ev)
        # N: batch size
        # S: source sequence length
        # L: target sequence length
        # E: embedding dimension of query and key
        # Ev: embedding dimension of value
        ((4, 5), (4, 5), (4, 3)),
        ((4, 2, 5), (4, 2, 5), (4, 2, 3)),
        # ((4, 1, 3, 5), (4, 3, 6, 5), (4, 2, 4, 2)),
    ],
)
def test_self_attention(size_q, size_k, size_v):
    query = torch.randn(size=size_q)
    key = torch.randn(size=size_k)
    value = torch.randn(size=size_v)

    expected_attention = torch.nn.functional.scaled_dot_product_attention(
        query.clone(), key.clone(), value.clone()
    )
    attention = scaled_dot_product_attention(query, key, value)

    assert torch.allclose(attention, expected_attention)
