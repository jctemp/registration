## Qualitative Analysis of Example Series: Irregular Respiratory Cycle

### Training Duration
- **GNCC**: 9 hours 30 minutes
- **GMI**: 10 hours 30 minutes
- **NCC**: 11 hours 30 minutes
- **MSE**: 9 hours 10 minutes

### Mean Square Error (MSE)
- **MSE**: 0.000887
- **NCC**: 0.00703
- **GNCC**: 0.000676
- **GMI**: 0.000556

The variance for MSE is negligible.

### Percentage of Negative Values in the Jacobian Determinant
- **GNCC** and **MSE**: Almost no negative values
- **NCC**: 0.00013
- **GMI**: 0.00072

For a 128x128 image, the variance is not negligible for all methods.

### Structural Similarity Index (SSIM)
- **MSE**: 0.89
- **NCC**: 0.90
- **GMI**: 0.91
- **GNCC**: 0.90

GNCC offers the best SSIM, followed by GMI, then NCC, and finally MSE. The variance is negligible.

### Comparison to Identity Function (No Warp)
- **MSE**: Halved, substantially less
- **SSIM**: 0.8787, indicating high similarity due to minimal deformations
- **Improvement**: Small for MSE (gaining 0.02), but other metrics achieve 0.9 and above

### Observations
- Better MSE and SSIM correlate with more negative values in the Jacobian determinant.
- Changes in MSE have a higher impact on the percentage of negative values than SSIM.

Overall, each measure exhibits distinct performance characteristics, with GMI and GNCC showing better overall metrics.

## Qualitative Analysis of Example Series: Irregular Respiratory Cycle

### Image Registration Target
- **Target**: Last image in the sequence
- **Focus**: Performance in regions with minimal deformation vs. maximal deformation

### Large Differences (Frames 80 to 90)
- **General Performance**: All models struggle
- **GNCC and GMI**: Better registration in the diaphragm region but introduce more deformations in the lung area, affecting blood vessels
- **NCC and MSE**: General problems with registration in these areas

### Small Dip/Rapid Change in Intensity (Frame 118)
- **Performance**: All models perform equally well except for MSE, which struggles

### Frames 180 to 190
- **GNCC and GMI**: Show good registration quality around the diaphragm region

### Rapid Change in Intensity (Frames 200 to 210)
- **Performance**: All models struggle with rapid changes from high to no difference in intensity
- **GMI**: Shows the best registration quality despite the rapid changes

### Minimal Differences to Last Image
- **NCC**: Shows the best performance in areas with little to no difference
- **GMI**: Occasionally exhibits good registration performance in these regions

### Observations
- **GNCC and GMI**: Perform better in regions with large deformations but introduce more deformations in the lung area.
- **NCC**: Best performance in regions with minimal deformations.
- **MSE**: Generally struggles, especially in regions with rapid intensity changes.
