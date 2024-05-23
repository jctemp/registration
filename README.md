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
- quantitative examination -> Jac Det. %, MSE, SSIM (mean, var)

## Construct loss function for lung registration

### Select `criteria_warped`:

| network    | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|--------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | mse-1  | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | ncc-1  | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1 | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |

```shell
for i in "mse-1" "ncc-1" "gmi-1" "gncc-1"; do
  ./batch transmorph --epochs 100 \
    --criteria_warped "${i}" --criteria_flow gl2d-1 \
    --registration_strategy soreg --registration_target last \
    --registration_depth 96 --registration_sampling 0 --registration_stride 1
done
```

### Use multiple criteria

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-1-ncc-1            | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1           | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |

```shell
for i in "gmi-1-ncc-1" "gncc-1-ncc-1" "gmi-0.5-gncc-0.5-ncc-1"; do
  ./batch transmorph --epochs 100 \
    --criteria_warped "${i}" --criteria_flow gl2d-1 \
    --registration_strategy soreg --registration_target last \
    --registration_depth 96 --registration_sampling 0 --registration_stride 1
done
```

### Select `criteria_flow`:

| network    | warped                 | flow      | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|-----------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-2    | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1    | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-0.5  | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | bel-1     | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | bel-0.375 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | bel-0.125 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |

```shell
for i in "gl2d-2" "gl2d-1" "gl2d-0.5" "bel-1" "bel-0.375" "bel-0.125"; do
  ./batch transmorph --epochs 100 \
    --criteria_warped gmi-0.5-gncc-0.5-ncc-1 --criteria_flow "${i}" \
    --registration_strategy soreg --registration_target last \
    --registration_depth 96 --registration_sampling 0 --registration_stride 1
done
```

## Investigate effect of `reg_depth`

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | last       | 64        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | last       | 128       | 1          | 0            | no       | adam      | 1e-4 |

```shell
for i in 32 64 96 128; do
  ./batch transmorph --epochs 100 \
    --criteria_warped gmi-0.5-gncc-0.5-ncc-1 --criteria_flow gl2d-1 \
    --registration_strategy soreg --registration_target last \
    --registration_depth "${i}" --registration_sampling 0 --registration_stride 1
done
```

## Reduce deformation by using mean target

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 0            | no       | adam      | 1e-4 |

```shell                                                                                                           
for i in last mean; do
  ./batch transmorph --epochs 100 \
    --criteria_warped gmi-0.5-gncc-0.5-ncc-1 --criteria_flow gl2d-1 \
    --registration_strategy soreg --registration_target "${i}" \
    --registration_depth 32 --registration_sampling 0 --registration_stride 1
done
```

## Effect of series sampling

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 1            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 4            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 8            | no       | adam      | 1e-4 |

```shell
for i in 1 2 4 8; do
  ./batch transmorph --epochs 100 \
    --criteria_warped gmi-0.5-gncc-0.5-ncc-1 --criteria_flow gl2d-1 \
    --registration_strategy soreg --registration_target mean \
    --registration_depth 32 --registration_sampling "${i}" --registration_stride 1
done
```

## Artificially add more variance in temporal dimension

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 1            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 4            | no       | adam      | 1e-4 |

```shell
for i in 1 2 4; do
  ./batch transmorph --epochs 100 \
    --criteria_warped gmi-0.5-gncc-0.5-ncc-1 --criteria_flow gl2d-1 \
    --registration_strategy soreg --registration_target mean \
    --registration_depth 32 --registration_sampling 1 --registration_stride "${i}"
done
```

## Identity loss to minimise unwanted deformations

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 1            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 1            | yes      | adam      | 1e-4 |

```shell                                                                                                           
./batch transmorph --epochs 100 \                                                                                  
  --criteria_warped gmi-0.5-gncc-0.5-ncc-1 --criteria_flow gl2d-1 \
  --registration_strategy soreg --registration_target mean \
  --registration_depth 32 --registration_sampling 1 --registration_stride 1 \
  --identity
```

## Adapting model capacity

| network          | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | identity | optimizer | lr   |
|------------------|--------|--------|--------------|------------|-----------|------------|----------|-----------|------|
| transmorph-huge  | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | yes      | adam      | 1e-4 |
| transmorph       | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | yes      | adam      | 1e-4 |
| transmorph-small | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | yes      | adam      | 1e-4 |
| transmorph-tiny  | gmi-1  | gl2d-1 | soreg        | last       | 32        | 1          | yes      | adam      | 1e-4 |

```shell
for i in transmorph-huge transmorph-small transmorph-tiny; do
  ./batch "${i}" --epochs 100 \
    --criteria_warped gmi-0.5-gncc-0.5-ncc-1 --criteria_flow gl2d-1 \
    --registration_strategy soreg --registration_target mean \
    --registration_depth 32 --registration_sampling 1 --registration_stride 1 
done
```
