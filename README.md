# Registration

```shell
. setup
```

## Hyperparameter

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

```shell
python -m reg generate \
  --network transmorph-identity \
  --criteria_warped zero-0 \
  --criteria_flow zero-0 \
  -o identity
  
 python -m reg batch --weight_directory identity identity.toml
```

## Construct loss function for lung registration

### Select `criteria_warped`:

| network    | warped | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|--------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | mse-1  | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | ncc-1  | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-1  | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1 | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped mse-1 \
  --criteria_flow gl2d-1 \
  --registration_strategy soreg \
  --registration_target last \
  --registration_depth 96 \
  --registration_stride 1 \
  --registration_sampling 0 \
  -o criteria_warped_config.toml
  
python -m reg batch --weight_directory test_criteria_warped criteria_warped_config.toml criteria_warped "mse-1" "ncc-1" "gmi-1" "gncc-1"

```

```shell
python -m reg eval --param criteria_warped --group_dir test_criteria_warped --idx 1
```

tmp/reg_t25sit2g
tmp/reg__ydli213
tmp/reg_m8sla1a7
tmp/reg_x9md7p39

### Use multiple criteria

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-1-ncc-1            | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1           | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped mse-1 \
  --criteria_flow gl2d-1 \
  --registration_strategy soreg \
  --registration_target last \
  --registration_depth 96 \
  --registration_stride 1 \
  --registration_sampling 0 \
  -o criteria_warped_config.toml
  
python -m reg batch --weight_directory test_mul_criteria_warped criteria_warped_config.toml criteria_warped "gmi-1-ncc-1" "gncc-1-ncc-1" "gmi-0.5-gncc-0.5-ncc-1"

```

### Select `criteria_flow`:

| network    | warped       | flow      | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|--------------|-----------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gncc-1-ncc-1 | gl2d-1    | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-0.5  | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-0.25 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | bel-1     | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | bel-0.5   | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | bel-0.25  | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-1 \
  --registration_strategy soreg \
  --registration_target last \
  --registration_depth 96 \
  --registration_stride 1 \
  --registration_sampling 0 \
  -o criteria_flow_config.toml
  
python -m reg batch --weight_directory test_criteria_flow criteria_flow_config.toml criteria_flow "gl2d-1" "gl2d-0.5" "gl2d-0.25" "bel-1" "bel-0.5" "bel-0.25"

```

## Investigate effect of `reg_depth`

| network    | warped       | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|--------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 64        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 96        | 1          | 0            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 128       | 1          | 0            | no       | adam      | 1e-4 |

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-0.5 \
  --registration_strategy soreg \
  --registration_target last \
  --registration_depth 96 \
  --registration_stride 1 \
  --registration_sampling 0 \
  -o registration_depth_config.toml
  
python -m reg batch --weight_directory test_registration_depth registration_depth_config.toml registration_depth 32 64 96 128

```

## Effect of series sampling

| network    | warped       | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|--------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 1            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 4            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 8            | no       | adam      | 1e-4 |

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-0.5 \
  --registration_strategy soreg \
  --registration_target last \
  --registration_depth 32 \
  --registration_stride 1 \
  --registration_sampling 0 \
  -o registration_sampling_config.toml
  
python -m reg batch --weight_directory test_registration_sampling registration_sampling_config.toml registration_sampling 1 2 4 8 16

```

## Reduce deformation by using mean target

| network    | warped       | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|--------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | last       | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | max        | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | min        | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | random     | 32        | 1          | 2            | no       | adam      | 1e-4 |

```shell                                                                                                           
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-1 \
  --registration_strategy soreg \
  --registration_target last \
  --registration_depth 32 \
  --registration_stride 1 \
  --registration_sampling 2 \
  -o registration_target_config.toml
  
python -m reg batch --weight_directory test_registration_target registration_target_config.toml registration_target "last" "mean" "max" "min" "random"

```

## Artificially add more variance in temporal dimension

| network    | warped       | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|--------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 2            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 2          | 2            | no       | adam      | 1e-4 |
| transmorph | gncc-1-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 4          | 2            | no       | adam      | 1e-4 |

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-1 \
  --registration_strategy soreg \
  --registration_target mean \
  --registration_depth 32 \
  --registration_stride 1 \
  --registration_sampling 2 \
  -o registration_stride_config.toml
  
python -m reg batch --weight_directory test_registration_stride registration_stride_config.toml registration_stride 1 2 4

```

## Identity loss to minimise unwanted deformations

| network    | warped                 | flow   | reg_strategy | reg_target | reg_depth | reg_stride | reg_sampling | identity | optimizer | lr   |
|------------|------------------------|--------|--------------|------------|-----------|------------|--------------|----------|-----------|------|
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 1            | no       | adam      | 1e-4 |
| transmorph | gmi-0.5-gncc-0.5-ncc-1 | gl2d-1 | soreg        | mean       | 32        | 1          | 1            | yes      | adam      | 1e-4 |

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-1 \
  --registration_strategy soreg \
  --registration_target mean \
  --registration_depth 32 \
  --registration_stride 1 \
  --registration_sampling 2 \
  -o identity_loss_config.toml
  
python -m reg batch --weight_directory test_identity_loss identity_loss_config.toml identity_loss true false

```

## Enforce temporal dependence in the loss function

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-1 \
  --registration_strategy soreg \
  --registration_target mean \
  --registration_depth 32 \
  --registration_stride 1 \
  --registration_sampling 2 \
  -o temporal_dependence_config.toml
  
python -m reg batch --weight_directory test_temporal_dependence temporal_dependence_config.toml

```

## Temporal dependence only

```shell
python -m reg generate \
  --network transmorph \
  --criteria_warped gncc-1-ncc-1 \
  --criteria_flow gl2d-1 \
  --context_length 256 \
  -o temporal_loss_only_config.toml
  
python -m reg batch --weight_directory temporal_loss_only temporal_loss_only_config.toml context_length 32 64 128 256

```

## Add additional image signal (fourier frequency)

TODO: image will have now two channels

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
