# Rebuttal experiment campaign summary

Completed result units: **100/100**.

## Rebuttal-ready conclusions

1. **The strongest positive result is spatially held-out Xenium breadth, not
   random-split performance.** Across 30 native-cell Xenium samples, the image
   model reached mean gene-macro Pearson 0.324 (10,000-bootstrap 95% CI
   0.277--0.368), versus 0.053 for spatial KNN, 0.046 for coordinates, and
   approximately 0 for the train mean. Sample-level paired Wilcoxon tests
   favored the image model over every control after Holm correction
   (all adjusted p <= 1.9e-8).
2. **The biological calibration is positive but should be described at the
   marker/pseudobulk level.** On train-selected marker/HVG overlap, the image
   model exceeded coordinate, KNN, and mean controls by 0.170, 0.172, and
   0.201 Pearson, respectively (all adjusted p <= 1.9e-8). Predicted-cell-type
   pseudobulk RMSE was 0.120 for images versus 0.187--0.224 for the controls
   (all adjusted p <= 7.5e-9).
3. **Same-organ transfer is better than a mean baseline but condition shift is
   a real limitation.** Across 18 same-organ targets, the image model improved
   gene-macro Pearson over the train mean by 0.170 (paired p = 7.6e-6).
   Normal-skin replication reached 0.359, whereas normal/melanoma transfers
   were only 0.063--0.094. The rebuttal should present this as evidence of
   within-context generalization plus a quantified domain-shift limitation.
4. **BLEEP does not improve the frozen-feature Ridge baseline here.** The
   official-algorithm adapter averaged 0.293 versus 0.324 for Ridge
   (difference -0.031; paired p = 1.4e-6). It is useful as a method-breadth
   control, not as a superiority claim.
5. **The current results do not support a broad claim that cell resolution is
   intrinsically more accurate than spot resolution.** The matched mean was
   0.365 at cell level, 0.365 at 8 µm, 0.363 at 16 µm, and 0.330 at 55 µm.
   Full-density HLCX022 was also nearly unchanged through 16 µm and decreased
   only from 0.365 to 0.361 at 55 µm, even though 55-µm bins mixed cell types
   extensively. The defensible claim is that coarsening changes target
   heterogeneity in a density-dependent way, not that single-cell prediction
   is universally more accurate.
6. **Visium HD-derived cell profiles require an explicit validity caveat.**
   About half of sampled CellViT polygons in the lung and pancreas examples
   received no canonical 2-µm bin assignment, versus 2.5% in mouse brain.
   Xenium results use native per-cell counts; Visium HD results use derived
   assignments and should not be described as equivalent ground truth.
7. **Generated captions are not independent biological validation.** Every
   readable caption explicitly contained the supplied cell-type label, and
   label recovery remained 99.9% after literal-label removal. Caption results
   should be disclosed as templated/label-conditioned artifacts, not used as
   leakage-free validation.

## Status by family

| family | n_complete | n_expected |
| --- | --- | --- |
| bleep_adapter | 30 | 30 |
| caption_audit | 1 | 1 |
| cross_sample | 18 | 18 |
| encoder_spatial | 7 | 7 |
| hd_assignment | 3 | 3 |
| pseudospot | 6 | 6 |
| skin_context | 5 | 5 |
| xenium_spatial | 30 | 30 |

## Reviewer concern coverage

| Concern | Evidence |
| --- | --- |
| Visium HD bin-to-cell validity | 3-dataset, 9,000-cell boundary/assignment audit |
| Breadth and heterogeneity | 30 Xenium samples; 17 organ labels; 25 conditions; 2 species |
| Local interpolation/autocorrelation | 4 contiguous edge holdouts, 5% spatial buffer, coordinate/KNN/mean controls |
| Low-correlation biological meaning | marker/HVG metrics and predicted-cell-type pseudobulk calibration |
| Cell versus spot resolution | 8/16/55 µm pseudo-spots plus full-density HLCX022 control |
| Method breadth and BLEEP clarity | 7 frozen encoders and an explicitly labeled official-algorithm BLEEP adapter |
| Generated text validity/leakage | caption label-mention and label-recovery audit |
| Context generalization | same-organ leave-one-sample-out and 5 normal/melanoma skin transfers |

Target provenance is separated explicitly: Xenium targets are native per-cell transcript counts, whereas Visium HD cell targets are derived by segmentation/bin assignment and are therefore reported with the assignment audit.

## Encoder spatial evaluation

Best mean gene-macro Pearson on HHDX011: **hoptimus0 = 0.307** (four contiguous edge holdouts).

## Xenium breadth

Macro mean gene-Pearson across all 30 samples: image=0.324, spatial_knn=0.053, coordinate=0.046, train_mean=0.000.

## BLEEP algorithm adapter

Mean gene-macro Pearson for the official-loss/top-50 BLEEP adapter across all 30 samples: **0.293**.

## Biological calibration

Mean image-model Pearson on train-selected marker/HVG overlap across available samples: **0.201**.

Mean predicted-cell-type pseudobulk RMSE (lower is better): image=0.120, spatial_knn=0.187, train_mean=0.188, coordinate=0.224.

## Cell versus pseudo-spot resolution

Subsample-matched mean gene-Pearson across available samples: cell_image=0.365, pseudo_spot_8um=0.365, pseudo_spot_16um=0.363, pseudo_spot_55um=0.330.

Full-density controls: HHDX011: cell_image=0.300, pseudo_spot_16um=0.300, pseudo_spot_55um=0.300, pseudo_spot_8um=0.300; HLCX022: cell_image=0.365, pseudo_spot_16um=0.365, pseudo_spot_55um=0.361, pseudo_spot_8um=0.365.

At 55 µm, full-density HLCX022 had 55.5--66.4% mixed-cell-type
spots containing 73.8--81.0% of evaluated cells across the four held-out
splits. HHDX011 was much sparser (3.5--4.1% mixed spots; 7.0--8.3% of cells),
showing why aggregation effects are density dependent.

## Visium HD assignment audit

Fraction of sampled CellViT cells receiving no valid 2 µm bin: d=50.6%, e=2.5%, g=50.0%.

## Generated-caption leakage audit

Audited 1070 readable captions; 100.0% explicitly contained the supplied cell-type label. Cell-type recovery after removing the literal label remained 0.999 accuracy.

## Interpretation constraints

- All target genes are selected from the training partition only.
- Spatial tests use contiguous edge holdouts plus a 5% coordinate-span buffer.
- External source-data inputs were treated as read-only.
- The BLEEP comparison is explicitly an algorithm adapter using the official projection/loss/retrieval design with frozen CONCH inputs, not a claim that the original spot-level checkpoint is cell-native.
- An ovarian age rerun and exact STBoost rerun remain unavailable unless their inaccessible/missing artifacts are supplied.
- Do not use the HD-derived cell targets, generated captions, BLEEP adapter, or
  pseudo-spot control to make stronger claims than the constraints above.
