import numpy as np
import pytest
import torch
from assertpy import assert_that
from dl.batch_norm.modules import BatchNorm


def test_single_forward_pass_no_training():
    batch_size = 4
    dim = 100_000
    batch = torch.randn((batch_size, dim)) * 2 + 1

    bn = BatchNorm(dim=dim)
    torch_bn = torch.nn.BatchNorm1d(
        num_features=dim, momentum=0, track_running_stats=True
    )

    y1 = bn(batch)
    y2 = torch_bn(batch)
    assert_that(torch.allclose(y1, y2, rtol=1e-05, atol=1e-05)).is_true()


def test_single_forward_pass_training():
    batch_size = 4
    dim = 100_000
    batch = torch.randn((batch_size, dim)) * 2 + 1

    bn = BatchNorm(dim=dim)
    bn.training = True
    torch_bn = torch.nn.BatchNorm1d(
        num_features=dim, momentum=0, track_running_stats=True
    )
    torch_bn.training = True

    y1 = bn(batch)
    y2 = torch_bn(batch)
    assert_that(torch.allclose(y1, y2, rtol=1e-05, atol=1e-05)).is_true()


@pytest.mark.parametrize("atol", [1 / 10 ** (5 + i) for i in range(5)])
def test_multiple_forward_passes_no_training(atol):
    batch_size = 4
    dim = 100_000
    batches = [torch.randn((batch_size, dim)) * 2 + 1 for _ in range(10)]

    bn = BatchNorm(dim=dim)
    torch_bn = torch.nn.BatchNorm1d(
        num_features=dim, momentum=0, track_running_stats=True
    )

    for batch in batches:
        y1 = bn(batch)
        y2 = torch_bn(batch)
        assert_that(torch.allclose(y1, y2, rtol=1e-05, atol=atol)).is_true()


# @pytest.mark.parametrize("execution_number", range(5))
@pytest.mark.parametrize("atol", [1 / 10 ** (5 + i) for i in range(5)])
def test_multiple_forward_passes_training(atol):
    batch_size = 4
    dim = 100_000
    batches = [torch.randn((batch_size, dim)) * 2 + 1 for _ in range(10)]

    bn = BatchNorm(dim=dim)
    bn.training = True
    torch_bn = torch.nn.BatchNorm1d(
        num_features=dim, momentum=0, track_running_stats=True
    )
    bn.training = False

    for batch in batches:
        y1 = bn(batch)
        y2 = torch_bn(batch)
        assert_that(torch.allclose(y1, y2, rtol=1e-05, atol=atol)).is_true()
