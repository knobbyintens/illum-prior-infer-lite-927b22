# illum-prior-infer-lite

Unofficial community reimplementation of the chroma-preserving, Retinex-guided
illumination-prior low-light enhancement idea. This repository ships an
**inference-only** wrapper that loads a third-party checkpoint and applies the
gated fusion step.

Scope and limitations:

- No training code is included; the model weights are downloaded from an
  external mirror.
- Experimental and unmaintained.
- Licensed under the GNU General Public License v3.0 (see `LICENSE`), which is
  incompatible with proprietary redistribution.

This project is not affiliated with Vela Imaging Labs.
