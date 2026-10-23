# Bohr–Sommerfeld Asymptotics for Limit-Circle Jacobi Matrices

John Zachary Fitch (zack@internetuniverse.org) · ORCID [0009-0007-7953-1531](https://orcid.org/0009-0007-7953-1531)

This repository holds the paper, its LaTeX sources, and the programs and stored outputs behind every numerical statement in it. The paper is [`spectral_limit_circle.pdf`](spectral_limit_circle.pdf), 79 pages, version of 22 October 2026 (release v1.1). Release v1.0 holds the first version, 68 pages; v1.1 is set in 12 pt, adds Section 14 on how the work began and how AI tools were used, and Appendix C on how the results were reached.

## What the paper proves

The paper gives explicit high-energy eigenvalue laws for every self-adjoint extension H_t of a zero-diagonal Jacobi matrix with links a_n = c n^α(1 + (κ + κ′(−1)^n)/n + O(n^{−1−δ})), α > 1, in the limit-circle range 2|κ′| < α − 1.

- **Balanced links (κ′ = 0).** It proves a Bohr–Sommerfeld law with an explicit offset and a rate. There is no quarter-period correction in the normalisation fixed at λ = 0.
- **Shifted powers.** It gives the second-order term.
- **Dimerised links (κ′ ≠ 0).** The far field is a Bessel problem of order ν = ½ − κ′/(α − 1). The two distinguished extensions move by ±κ′/(2(α − 1)), and every other extension follows one of them.

Applications include squeezing operators, birth–death processes with polynomial rates, and Berg and Szwarc's critical indices.

## Layout

- `spectral_limit_circle.tex`, `sec*.tex`, `app_*.tex` and `refs.tex` are the LaTeX sources.
- `scripts/` holds the programs named in Appendix B of the paper.
- `outputs/` holds their stored outputs, including the per-eigenvalue truncation data behind Table 3 and Figure 2.
- `SHA256SUMS.txt` lists a SHA-256 checksum for every file above. Verify with `sha256sum -c SHA256SUMS.txt`.

## Reproducing the checks

Run these from the repository root. They need Python 3, mpmath and SymPy.

```
python scripts/exact_checks.py
python scripts/p2_reduction.py
python scripts/airy_envelopes.py
python scripts/varpi_bd.py
python scripts/bv_closed_forms.py
python scripts/qfamily_indices.py
python scripts/conjH.py
```

The eigenvalue runs (`validate_sturm.py`, `truncation_run.py`, `diag_run.py` and `diag_balanced.py`) need CuPy and a CUDA device. `sturm.py` also contains the NumPy reference count. `make_table.py` rebuilds Table 3 from the stored data.

## Not included

- **The typesetting kit.** The sources use the author's `etch-math` and `etch-plot` style files, and `scripts/fig_laws.py` imports the kit's `es_figure` module. These are not part of this repository, so the LaTeX does not compile as-is. The compiled PDF is provided.

## Companion paper

[The Volume Gap at Five-Valent Vertices](https://github.com/johnzfitch/five-valent-volume-gap) comes from the same operator with su(2) labels.

## Licence

The paper (the LaTeX sources and the PDF) and the stored outputs are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code (all `.py` files) is licensed under the MIT License. See [`LICENSE`](LICENSE).
