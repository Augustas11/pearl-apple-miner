"""mlx-lm model file for pearl2mlx *exact* (A2) checkpoints.

pearl2mlx copies this file next to the converted weights and sets
``"model_file": "pearl_layers.py"`` in config.json. Stock ``mlx_lm.load``
(and therefore mlx_lm.generate / perplexity / benchmark / evaluate / server)
honours ``model_file`` and builds the model from the ``Model`` / ``ModelArgs``
defined here, so no wrapper or monkey-patch is needed.

The only difference from mlx_lm's Llama: every module path listed in
``config["pearl_smooth"]`` (int7 layers whose SmoothQuant scale could not be
folded into a preceding RMSNorm, e.g. every o_proj) becomes a
``PearlQuantizedLinear`` that computes ``quantized_matmul(x * smooth, W)``.

No `from __future__ import annotations` here: mlx_lm loads this file via
spec_from_file_location without registering it in sys.modules, which breaks
@dataclass string-annotation resolution.

Swift mirror (mlx-swift-lm, ParoQuant pattern; not implemented here):
  * Loader: when config.json has ``pearl_smooth``, after building the Llama
    model and before ``quantize(model:...)``, replace each listed Linear with a
    ``PearlQuantizedLinear: QuantizedLinear`` that owns ``smooth: MLXArray``
    (shape [in]) and overrides ``callAsFunction(x)`` to
    ``quantizedMM(x * smooth, weight, scales, biases, transpose: true,
    groupSize, bits)``; the per-layer group sizes come from the standard
    ``quantization`` map that Load.swift already applies.
  * Weight key: ``<path>.smooth`` (bf16, [in]) is loaded like any parameter.
"""

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import llama


@dataclass
class ModelArgs(llama.ModelArgs):
    pearl_smooth: Optional[List[str]] = None


class PearlQuantizedLinear(nn.QuantizedLinear):
    """QuantizedLinear with a per-input-channel SmoothQuant multiply."""

    def __init__(self, input_dims, output_dims, bias=True, group_size=None,
                 bits=None, mode="affine"):
        super().__init__(input_dims, output_dims, bias, group_size, bits, mode)
        self.smooth = mx.ones((input_dims,))
        self.freeze()

    def __call__(self, x):
        return super().__call__(x * self.smooth)


class PearlLinear(nn.Linear):
    """Pre-quantization placeholder; nn.quantize turns it into PearlQuantizedLinear."""

    def __init__(self, input_dims, output_dims, bias=False):
        super().__init__(input_dims, output_dims, bias=bias)
        self.smooth = mx.ones((input_dims,))

    def __call__(self, x):
        return super().__call__(x * self.smooth)

    def to_quantized(self, group_size=None, bits=None, mode="affine",
                     quantize_input=False):
        if quantize_input:
            raise ValueError("PearlLinear does not support activation quantization")
        out_dims, in_dims = self.weight.shape
        return PearlQuantizedLinear(in_dims, out_dims, "bias" in self,
                                    group_size, bits, mode)


class Model(llama.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        for path in args.pearl_smooth or []:
            *parents, leaf = path.split(".")
            mod = self
            for p in parents:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            old = getattr(mod, leaf)
            out_dims, in_dims = old.weight.shape
            setattr(mod, leaf, PearlLinear(in_dims, out_dims, bias="bias" in old))
