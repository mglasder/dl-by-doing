import numpy as np
import pytest
import torch
from assertpy import assert_that
from dl.batch_norm.modules import BatchNorm


@pytest.mark.parametrize("atol", [1 / 10 ** (3 + i) for i in range(5)])
@pytest.mark.parametrize("train", [True, False])
def test_single_forward_pass(train, atol):
    batch_size = 4
    dim = 100_000
    batch = torch.randn((batch_size, dim)) * 2 + 1

    bn = BatchNorm(dim=dim, momentum=0.1)
    torch_bn = torch.nn.BatchNorm1d(
        num_features=dim,
        momentum=0.1,
        track_running_stats=True,
    )
    bn.train(train)
    torch_bn.train(train)

    y1 = bn(batch)
    y2 = torch_bn(batch)
    assert_that(torch.allclose(y1, y2, rtol=1e-05, atol=atol)).is_true()


@pytest.mark.parametrize("atol", [1 / 10 ** (3 + i) for i in range(5)])
def test_multiple_forward_passes_training(atol):
    # TODO: test fails, find out why
    batch_size = 4
    dim = 100_000
    batches = [torch.randn((batch_size, dim)) * 2 + 1 for _ in range(10)]

    bn = BatchNorm(dim=dim, momentum=0.1)
    torch_bn = torch.nn.BatchNorm1d(
        num_features=dim, momentum=0.1, track_running_stats=True
    )

    # "train" the bn layer (without actually training the parameters)
    bn.train(True)
    torch_bn.train(True)
    for batch in batches:
        _ = bn(batch)
        _ = torch_bn(batch)

    # switch to inference
    bn.train(False)
    torch_bn.train(False)
    y1 = bn(batches[-1])
    y2 = torch_bn(batches[-1])
    assert_that(torch.allclose(y1, y2, rtol=1e-05, atol=atol)).is_true()
