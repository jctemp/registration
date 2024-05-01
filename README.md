# Registration

- network
- criteria_warped
- criteria_flow
- registration_strategy
- registration_target
- registration_depth
- registration_stride
- identity_loss
- optimizer
- learning_rate

## Metrics to evaluate experiments

- qualitative examination -> warped, flow, difference
- quantitative examination -> Jac Det. %, MSE, SSIM (mean, var, min, max)

MILESTONE: PREFUL pipeline output

## Construct loss function for lung registration

### Select `criteria_warped`:

| network    | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | optimizer | lr   |
|------------|--------|--------|--------------|------------|-----------|------------|-----------|------|
| transmorph | mse-1  | gl2d-1 | soreg        | last       | 128       | 1          | adam      | 1e-4 |
| transmorph | ncc-1  | gl2d-1 | soreg        | last       | 128       | 1          | adam      | 1e-4 |
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 128       | 1          | adam      | 1e-4 |

### Select `criteria_flow`:

| network    | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | optimizer | lr   |
|------------|--------|--------|--------------|------------|-----------|------------|-----------|------|
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 128       | 1          | adam      | 1e-4 |
| transmorph | gmi-1  | bel-1  | soreg        | last       | 128       | 1          | adam      | 1e-4 |

### Use multiple criteria

| network    | warped       | flow   | reg_strategy | reg_target | reg_depth | reg_stride | optimizer | lr   |
|------------|--------------|--------|--------------|------------|-----------|------------|-----------|------|
| transmorph | gmi-1-ssim-1 | gl2d-1 | soreg        | last       | 128       | 1          | adam      | 1e-4 |
| transmorph | gmi-1-ncc-1  | gl2d-1 | soreg        | last       | 128       | 1          | adam      | 1e-4 |

## Investigate effect of `reg_depth`

| network    | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | optimizer | lr   |
|------------|--------|--------|--------------|------------|-----------|------------|-----------|------|
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | adam      | 1e-4 |
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 64        | 1          | adam      | 1e-4 |
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 96        | 1          | adam      | 1e-4 |
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 128       | 1          | adam      | 1e-4 |

### Modify training: only present a segment to the model instead of complete series

| network    | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | optimizer | lr   |
|------------|--------|--------|--------------|------------|-----------|------------|-----------|------|
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | adam      | 1e-4 |

NOTE: this is for runtime optimisation regarding the training

## Reduce deformation by using mean target

| network    | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | optimizer | lr   |
|------------|--------|--------|--------------|------------|-----------|------------|-----------|------|
| transmorph | gmi-1  | gl2d-1 | soreg        | mean       | 32        | 1          | adam      | 1e-4 |

NOTE: Helps to minimise required deformation leading to smaller and more accurate deformations and reducing difference

## Adapting model capacity

| network          | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | optimizer | lr   |
|------------------|--------|--------|--------------|------------|-----------|------------|-----------|------|
| transmorph-huge  | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | adam      | 1e-4 |
| transmorph       | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | adam      | 1e-4 |
| transmorph-small | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | adam      | 1e-4 |
| transmorph-tiny  | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | adam      | 1e-4 |

## Identity loss to minimise unwanted deformations

