# Python examples 

In this directory, we provide a couple of examples, described below.

## Walkthrough tutorial

[WavAugment_walkthrough.ipynb](./examples/python/WavAugment_walkthrough.ipynb) [(open in colab)](https://colab.research.google.com/github/facebookresearch/WavAugment/blob/master/examples/python/WavAugment_walkthrough.ipynb) provides a succint walkthrough tutorial, showing how effects can be applied on a piece of speech (recorded over the mic or pre-recorded).

## Processing a single file

The script [process_file.py](./process_file.py) gives a taste on how different compositions of speech augmentation techniques sound like, by allowing to augment single files.  It suppors a few randomized data augmentations: pitch, reverberation, temporal masking, band rejection, and clipping.

A typical usage is:
```bash
python process_file.py --input_file=./tests/test.wav \ 
    --output_file=augmented.wav \ 
    --chain=pitch,reverb,time_drop \ 
    --pitch_shift_max=500 \ 
    --t_ms=100
```
where `--chain` specifies a list of augmentations applied sequentially, left-to-right; `t_ms` and `pitch_shift_max` specify parameters of the augmentations. `augmented.wav` would contain the randomly augmented sound.


## Usage in self-supervised learning

In [librispeech_selfsupervised.py](./librispeech_selfsupervised.py) we use WavAugment in a way that can be used for self-supervised learning.
We define a dataset that iterates over Librispeech data, reads a (randomly shifted) sequence of pre-defined length from each file 
and returns two copies of it, independently augmented in different ways. This example does not learns a model, only measures the dataset reading time.

The code is thoroughly documented. This command will download `dev-clean` in the `./data` directory (if needed), iterate over it,
extracting sequences of 1 second length. The batches of size 32 would be prepared by 8 DataLoader workers.

```bash
python librispeech_selfsupervised.py --data=./data --subset=dev-clean --sequence_length_seconds=1 --n_workers=8 --download --batch_size=8
```

Iterating over Librispeech train-clean-100 (100 hours of audio) with 16 workers takes 2 seconds without any data augmentation. 
With WavAugment data augmentation it takes around 5 seconds (on a solid server).
