# Conserving mass, momentum, and energy for the Benjamin-Bona-Mahony, Korteweg-de Vries, and nonlinear Schrödinger equations

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17936837.svg)](https://zenodo.org/doi/10.5281/zenodo.17936837)

This repository contains information and code to reproduce the results presented
in the article
```bibtex
@online{ranocha2025conserving,
  title={Conserving mass, momentum, and energy for the {B}enjamin-{B}ona-{M}ahony,
         {K}orteweg-de {V}ries, and nonlinear {S}chr{\"o}dinger equations},
  author={Ranocha, Hendrik and Ketcheson, David I},
  year={2025},
  month={12},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{ranocha2025conservingRepro,
  title={Reproducibility repository for
         "{C}onserving mass, momentum, and energy for the {B}enjamin-{B}ona-{M}ahony,
         {K}orteweg-de {V}ries, and nonlinear {S}chr{\"o}dinger equations"},
  author={Ranocha, Hendrik and Ketcheson, David I},
  year={2025},
  howpublished={\url{https://github.com/ranocha/2025_BBM_KdV_NLS}},
  doi={10.5281/zenodo.17936837}
}
```


## Abstract

Many important partial differential equations (PDEs) possess multiple invariant
quantities whose values characterize solutions in an essential way. Specially
designed (structure-preserving) numerical methods are capable of preserving one
or sometimes two invariants, often corresponding to mass, momentum, or energy.
However, many important systems possess additional invariants, and completely
integrable systems possess an infinite number of them.
We propose and study a class of numerical discretizations that are capable of
preserving several polynomial invariants. In space, we use Fourier Galerkin methods,
while in time we use a combination of orthogonal projection and relaxation. We prove
and numerically demonstrate the conservation properties of the method by applying it
to the Benjamin-Bona-Mahoney, Korteweg-de Vries, and nonlinear Schrödinger (NLS) PDEs
as well as a hyperbolic approximation of NLS. For each of these equations, we are
able to conserve mass, momentum, and energy up to numerical precision. We show that
this conservation leads to reduced growth of numerical errors for long-term simulations.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/).
The numerical experiments presented in this article were performed using
Julia v1.10.10.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory of this repository and follow the instructions
described in the `README.md` file therein.


## Authors

- [Hendrik Ranocha](https://ranocha.de) (Johannes Gutenberg University Mainz, Germany)
- David I. Ketcheson (KAUST, Saudi Arabia)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
