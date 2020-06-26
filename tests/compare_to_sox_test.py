# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import warnings
import tempfile
import subprocess
import torchaudio
import pathlib
import numpy as np

import augment

test_wav = pathlib.Path(__file__).parent / 'test.wav'
assert test_wav.exists()


def run_sox_command(test_wav, cl_parameters):
    with tempfile.NamedTemporaryFile(suffix='.wav') as t_file:
        output_name = t_file.name
        res = subprocess.run(
            ['sox', str(test_wav), output_name] + cl_parameters
        )
        assert res.returncode == 0

        y, sr = torchaudio.load(output_name)
    return y, sr


def apply_chain(test_wav, chain):
    x, sr = torchaudio.load(test_wav)

    assert sr == 16000

    src_info = {'channels': x.size(0),
                'length': x.size(1),
                'precision': 32,
                'rate': 16000.0,
                'bits_per_sample': 32}

    target_info = {'channels': 1,
                   'length': 0,
                   'precision': 32,
                   'rate': 16000.0,
                   'bits_per_sample': 32}

    y = chain.apply(
        x, src_info=src_info, target_info=target_info)
    return y

def test_pitch():
    y1, _ = run_sox_command(test_wav, ["pitch", "-100"])

    chain = augment.EffectChain().pitch(-100).rate(16000)
    y2 = apply_chain(test_wav, chain)

    assert y1.size() == y2.size()

    # NB: higher tolerance due to all the discretization done on save/load
    assert torch.allclose(y1, y2, rtol=1e-4, atol=1e-4)


def test_reverb():
    y1, _ = run_sox_command(test_wav, ["reverb", "50", "50", "100"])

    chain = augment.EffectChain().reverb(50, 50, 100).channels()
    y2 = apply_chain(test_wav, chain)

    assert y1.size() == y2.size()

    # NB: higher tolerance due to all the discretization done on save/load
    assert torch.allclose(y1, y2, rtol=1e-4, atol=1e-4)


def test_bandreject():
    y1, _ = run_sox_command(test_wav, ["sinc", "-a", "120", "2000-1000"])

    chain = augment.EffectChain().sinc("-a", "120", "2000-1000")
    y2 = apply_chain(test_wav, chain)

    assert y1.size() == y2.size()

    # NB: higher tolerance due to all the discretization done on save/load
    assert torch.allclose(y1, y2, rtol=1e-4, atol=1e-4)
