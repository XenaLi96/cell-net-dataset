# sMMC-22M rebuttal experiment package

This double-blind package contains the evaluation code and result artifacts
used for the rebuttal experiments. It intentionally excludes author names,
institutional information, machine-specific paths, host names, scheduler
configuration, raw execution logs, job identifiers, and source-data files.

## Package contents

- `code/`: portable data preparation, validation, evaluation, auditing, and
  aggregation scripts.
- `results/RESULTS_SUMMARY.md`: concise interpretation of the completed
  experiments, including negative findings and claim limitations.
- `results/tables/`: detailed run-level, sample-level, aggregate, uncertainty,
  and paired-test CSV files.
- `results/figures/`: metadata-stripped PNG versions of the aggregate figures.
- `METHODS_AND_REPRODUCIBILITY.md`: protocol, input schemas, and script map.
- `ANONYMIZATION.md`: the double-blind exclusions and validation procedure.
- `requirements.txt`: versions used for the reported analyses.
- `FILES.sha256`: integrity checksums for all other package files.

## Completed experiment units

| Family | Completed |
| --- | ---: |
| Spatially held-out Xenium breadth | 30/30 samples |
| Same-organ cross-sample transfer | 18/18 targets |
| Pseudo-spot resolution control | 6/6 samples |
| Skin context shift | 5/5 contrasts |
| BLEEP algorithm adapter | 30/30 samples |
| Encoder spatial comparison | 7/7 encoders |
| Visium HD assignment audit | 3/3 datasets |
| Generated-caption audit | 1/1 collection |

The 100 expected result units are complete. Target provenance is handled
explicitly: Xenium analyses use native per-cell transcript counts, whereas
Visium HD cell profiles are derived through segmentation and bin assignment.

## Primary findings

- Across 30 spatially held-out Xenium samples, mean gene-macro Pearson was
  0.324 for the image model, 0.053 for spatial KNN, 0.046 for coordinates,
  and approximately zero for the training mean. Image-versus-control paired
  tests remained significant after Holm correction.
- Marker/HVG correlations and predicted-cell-type pseudobulk calibration
  favored the image model over spatial and mean controls.
- Same-organ transfer was positive on average, while normal/melanoma transfer
  showed a substantial condition-domain shift.
- The BLEEP algorithm adapter did not improve the frozen-feature Ridge
  baseline in this protocol.
- Pseudo-spot controls do not establish a universal accuracy advantage for
  cell resolution; they show density-dependent target mixing under coarsening.
- Visium HD assignment reliability varied by dataset, and generated captions
  showed strong label-conditioning leakage. These are reported as limitations,
  not as positive validation results.

## Reproduction scope

The scripts are provided for methods inspection and rerunning with legally
obtained source data and model features. Raw spatial-transcriptomics data,
histology images, foundation-model weights, and large embedding caches are not
redistributed in this package. All input and output locations are supplied at
runtime; no author- or server-specific path is embedded in the package.

Start with `METHODS_AND_REPRODUCIBILITY.md` and the `--help` option of each
script. Numerical results can be checked directly from `results/tables/`.
