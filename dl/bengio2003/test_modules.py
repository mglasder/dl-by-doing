import torch
from dl.bengio2003.modules import NPLM, NPLMref
from assertpy import assert_that
import pytest


@pytest.fixture
def config():
    return {
        "hidden_units": 6,
        "context": 3,
        "features": 2,
        "vocab_size": 10,
        "batch_size": 32,
        "g": torch.Generator().manual_seed(42),
    }


@pytest.fixture
def NPLM(config):

    return NPLM(
        batch_size=config["batch_size"],
        hidden_units=config["hidden_units"],
        context=config["context"],
        features=config["features"],
        vocab_size=config["vocab_size"],
        g=config["g"],
    )


@pytest.fixture
def NPLMref(config):
    return NPLMref(
        block_size=config["context"],
        vocab_size=config["vocab_size"],
        n_embd=config["features"],
        n_embd2=config["hidden_units"],
    )


def test_NPLM_and_NPLMref_have_same_output_dims(NPLM, NPLMref, config):

    nplm = NPLM
    nplm_adv = NPLMref

    x = torch.randint(
        size=(config["batch_size"], config["context"], config["features"]),
        low=0,
        high=config["vocab_size"] - 1,
    )

    out1 = nplm.forward(x)
    out2 = nplm_adv.forward(x)

    assert_that(out1.shape).is_equal_to(out2.shape)


def test_NPLM_and_NPLMref_have_same_n_parameters(NPLM, NPLMref):

    nplm = NPLM
    nplm_adv = NPLMref

    sum1 = 0
    for p in nplm.parameters():
        sum1 += p.nelement()

    sum2 = 0
    for p in nplm_adv.parameters():
        sum2 += p.nelement()

    assert_that(sum1).is_equal_to(sum2)
