Both criteria are evaluated at their **proven optimal** thresholds, so the
optimizer plays no part in any disagreement -- it is the metric, not the search.

| Image | C | Dice picks | Dice margin | PSNR picks | SSIM picks | FSIM picks |
|---|---|---|---|---|---|---|
| phantom_c2 | 2 | otsu | 0.0002 | otsu | otsu | **kapur** |
| phantom_c3 | 3 | otsu | 0.3993 | otsu | otsu | otsu |
| phantom_c4 | 4 | otsu | 0.0006 | otsu | otsu | **kapur** |
| phantom_c5 | 5 | otsu | 0.0004 | otsu | **kapur** | **kapur** |
| phantom_c6 | 6 | otsu | 0.0020 | otsu | **kapur** | **kapur** |

Bold marks a metric selecting a different criterion than Dice.

| Metric | Disagrees with Dice |
|---|---|
| PSNR | 0 of 5 |
| SSIM | 2 of 5 |
| FSIM | 4 of 5 |

### Reading this honestly

- Where the two segmentations genuinely differ (Dice margin > 0.05, 1 case), every metric agrees with Dice in 1 of them.
- Where they are near-identical (Dice margin <= 0.05, 4 cases), the metrics disagree freely -- they are tie-breaking between segmentations that are practically the same.

So the conventional metrics are not simply wrong. They track segmentation quality when the difference is large, and become arbitrary when it is small -- which is exactly the regime where papers report them to separate competing methods.

