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

def test_empty_chain():
    x = torch.arange(0, 8000).float()

    src_info = {'channels': 1,
                'length': x.size(0),
                'precision': 32,
                'rate': 16000.0,
                'bits_per_sample': 32}

    target_info = {'channels': 1,
                   'length': 0,
                   'precision': 32,
                   'rate': 16000.0,
                   'bits_per_sample': 32}

    y = augment.EffectChain().apply(
        x, src_info=src_info, target_info=target_info)

    assert x.view(-1).allclose(y.view(-1))


def test_non_empty_chain():
    x, sr = torchaudio.load(test_wav)

    src_info = {'channels': 1,
                'length': x.size(1),
                'precision': 32,
                'rate': 16000.0,
                'bits_per_sample': 32}

    target_info = {'channels': 1,
                   'length': 0,
                   'precision': 32,
                   'rate': 16000.0,
                   'bits_per_sample': 32}

    effects = augment.EffectChain().bandreject(1, 20000)

    y = effects.apply(x, src_info=src_info,
                          target_info=target_info)

    assert x.size() == y.size(), f'{y.size()}'
    assert not x.allclose(y)

def convert_pitch_cl(test_wav):
    with tempfile.NamedTemporaryFile(suffix='.wav') as t_file:
        output_name = t_file.name
        res = subprocess.run(
            ['sox', str(test_wav), output_name, 'pitch', '100'])
        assert res.returncode == 0

        y, sr = torchaudio.load(output_name)
    return y, sr


def convert_pitch_augment(test_wav):
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

    y = augment.EffectChain().pitch(100).rate(16000).apply(
        x, src_info=src_info, target_info=target_info)
    return y, sr

def test_agains_cl():
    y1, _ = convert_pitch_cl(test_wav)
    y2, _ = convert_pitch_augment(test_wav)

    assert y1.size() == y2.size()

    # NB: higher tolerance due to all the discretization done on save/load
    assert torch.allclose(y1, y2, rtol=1e-3, atol=1e-3)

    # just to make sure something is happening
    x, sr = torchaudio.load(test_wav)
    assert not torch.allclose(x, y2, rtol=1e-3, atol=1e-3)

def test_stochastic_pitch():
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

    def random_pitch(): return np.random.randint(100, 500)
    y = augment.EffectChain().pitch(random_pitch).rate(16000).apply(
        x, src_info=src_info, target_info=target_info)
    assert not torch.allclose(x, y, rtol=1e-3, atol=1e-3)

    
def test_additive_noise():
    x, sr = torchaudio.load(test_wav)

    noise = torch.zeros_like(x)

    src_info = {'channels': 1,
                'length': x.size(1),
                'precision': 32,
                'rate': 16000.0,
                'bits_per_sample': 32}

    target_info = {'channels': 1,
                   'length': 0,
                   'precision': 32,
                   'rate': 16000.0,
                   'bits_per_sample': 32}

    y = augment.EffectChain() \
            .additive_noise(noise_generator=lambda: x, snr=10.0) \
            .apply(x, src_info=src_info, target_info=target_info)

    assert torch.allclose(x, y)

def test_number_effects():
    assert len(augment.EffectChain.KNOWN_EFFECTS) == 61
