# WavAugment

WavAugment performs data augmentation on audio data. 
The audio data is represented as [pytorch](https://pytorch.org/) tensors. 

It is particularly useful for speech data. 
Among others, it implements the augmentations that we found to be most useful for self-supervised learning 
(_Data Augmenting Contrastive Learning of Speech Representations in the Time Domain_, E. Kharitonov, M. Rivière, G. Synnaeve, L. Wolf, P.-E. Mazaré, M. Douze, E. Dupoux. [[arxiv]](https://arxiv.org/abs/2007.00991)):

* Pitch randomization,
* Reverberation,
* Additive noise,
* Time dropout (temporal masking),
* Band reject,
* Clipping

Internally, WavAugment uses [libsox](http://sox.sourceforge.net/libsox.html) and allows interleaving of libsox- and pytorch-based effects. 

### Requirements
 * Linux (WavAugment is not tested under MacOS and might not work properly);
 * [pytorch](pytorch.org) (however, there is also an option of using WavAugment directly from C++ w/o torch, see below);
 * `libsox`: if you have [torchaudio](https://github.com/pytorch/audio) installed, most likely you already have `libsox`. Otherwise, you need to install it, e.g. by running `sudo apt-get install sox libsox-dev libsox-fmt-all`

### Installation
To install WavAugment, run the following command:
```bash
git clone git@github.com:facebookresearch/WavAugment.git && cd WavAugment && python setup.py develop
```

### Testing
Requires pytest (`pip install pytest`)

```bash
python -m pytest -v --doctest-modules
```

## Usage

First of all, we provide thouroughly documented [examples](./examples/python), where we demonstrate how a data-augmented dataset interface works.

### The `EffectChain`

The central object is the chain of effects, `EffectChain`, that are applied on a `torch.Tensor` to produce another `torch.Tensor`. 
This chain can have multiple effects composed:
```python
import augment
effect_chain = augment.EffectChain().pitch(100).rate(16_000)
```
Parameters of the effect coincide with those of libsox (http://sox.sourceforge.net/libsox.html); however, you can also randomize the parameters by providing a python `Callable` and mix them with standard parameters:
```python
import numpy as np
random_pitch_shift = lambda: np.random.randint(-100, +100)
# the pitch will be changed by a shift somewhere between (-100, +100)
effect_chain = augment.EffectChain().pitch("-q", random_pitch_shift).rate(16_000)
```
Here, the flag`-q` makes `pitch` run faster at some expense of the quality.
If some parameters are provided by a Callable, this Callable will be invoked every time `EffectChain` is applied (eg. to generate random parameters).

### Applying the chain

To apply a chain of effects on a torch.Tensor, we code the following:
```python
output_tensor = augment.EffectChain().pitch(100).rate(16_000).apply(input_tensor, \
    src_info=src_info, target_info=target_info)
```
WavAugment expects `input_tensor` to have a shape of (channels, length). As `input_tensor` does not contain important meta-information, such as sampling rate, we need to provide it manually.
This is done by passing two dictionaries, `src_info` (meta-information about the input format) and `target_info` (our expectated format for the output).

At minimum, we need to set the sampling rate for the input tensor: `{'rate': 16_000}`. 

### Example usage

Below is a small gist of a potential usage:

```python
import augment
import numpy as np

x, sr = torchaudio.load(test_wav)

# input signal properties
src_info = {'rate': sr}

# output signal properties
target_info = {'channels': 1, 
               'length': 0, # not known beforehand
               'rate': 16_000}
# write down the chain of effects with their string parameters and call .apply()
# effects are specified as a chain of method calls with parameters that can be 
# strings, numbers, or callables. The latter case is used for generating randomized
# transformations
random_pitch = lambda: np.random.randint(-400, -200)
y = augment.EffectChain().pitch(random_pitch).rate(16_000).apply(x, \
    src_info=src_info, target_info=target_info)
```

## Important notes
It often happens that a command-line invocation of sox would change effect chain. To get a better idea of what sox executes internally, you can launch it with a -V flag, eg by running:
 ```bash
sox -V tests/test.wav out.wav reverb 0 50 100
```
we will see something like:
```
sox INFO sox: effects chain: input        16000Hz  1 channels
sox INFO sox: effects chain: reverb       16000Hz  2 channels
sox INFO sox: effects chain: channels     16000Hz  1 channels
sox INFO sox: effects chain: dither       16000Hz  1 channels
sox INFO sox: effects chain: output       16000Hz  1 channels
```
This output tells us that the `reverb` effect changes the number of channels, which are squashed into 1 channel by the `channel` effect. Sox also added `dither` effect to hide processing artifacts.

WavAugment remains explicit and doesn't add effects under the hood. 
If you want to emulate a Sox command that decomposes into several effects, we advise to consult `sox -V` and apply the effects manually. 
Try it out on some files before running a heavy machine-learning job. 

## But I want to use it from C++
### Installation
It is possible to use directly WavAugment's C++ interface to libsox.
You will need to install `libsox`, e.g. by running
```bash
sudo apt-get install sox libsox-dev libsox-fmt-all
```
The C++ interface is provided as a single-header library, so you only need to include [this file](./augment/speech_augment.h).

## Citation
If you find WavAugment useful in your research, please consider citing:
```
@article{wavaugment2020,
  title={Data Augmenting Contrastive Learning of Speech Representations in the Time Domain},
  author={Kharitonov, Eugene and Rivi{\`e}re, Morgane and Synnaeve, Gabriel and Wolf, Lior and Mazar{\'e}, Pierre-Emmanuel and Douze, Matthijs and Dupoux, Emmanuel},
  journal={arXiv preprint arXiv:2007.00991},
  year={2020}
}
```

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
WavAugment is MIT licensed, as found in the LICENSE file.

