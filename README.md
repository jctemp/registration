# Registration

- network
- criteria_warped
- criteria_loss
- optimizer
- learning_rate
- registration_strategy
- registration_target
- identity_loss

## Constructing a loss function:

| model      | max_epoch | series_len | img_loss_fn | flow_loss_fn | target | data_mod |
|------------|-----------|:-----------|-------------|--------------|--------|----------|
| transmorph | 100       | 32         | mse:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 64         | mse:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 128        | mse:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 192        | mse:1       | gl2d:1       | last   | norm     |

| model      | max_epoch | series_len | img_loss_fn | flow_loss_fn | target | data_mod |
|------------|-----------|------------|-------------|--------------|--------|----------|
| transmorph | 100       | 32         | ncc:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 64         | ncc:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 128        | ncc:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 192        | ncc:1       | gl2d:1       | last   | norm     |

| model      | max_epoch | series_len | img_loss_fn | flow_loss_fn | target | data_mod |
|------------|-----------|------------|-------------|--------------|--------|----------|
| transmorph | 100       | 32         | gmi:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 64         | gmi:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 128        | gmi:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 192        | gmi:1       | gl2d:1       | last   | norm     |

| model      | max_epoch | series_len | img_loss_fn | flow_loss_fn | target | data_mod |
|------------|-----------|------------|-------------|--------------|--------|----------|
| transmorph | 100       | 32         | gmi:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 64         | gmi:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 128        | gmi:1       | gl2d:1       | last   | norm     |
| transmorph | 100       | 192        | gmi:1       | gl2d:1       | last   | norm     |

| model      | max_epoch | series_len | img_loss_fn | flow_loss_fn | target | data_mod |
|------------|-----------|------------|-------------|--------------|--------|----------|
| transmorph | 300       | 128        | mse:1       | gl2d:1       | last   | norm     |
| transmorph | 300       | 128        | ncc:1       | gl2d:1       | last   | norm     |
| transmorph | 300       | 128        | gmi:1       | gl2d:1       | last   | norm     |

| model      | max_epoch | series_len | img_loss_fn  | flow_loss_fn | target | data_mod |
|------------|-----------|------------|--------------|--------------|--------|----------|
| transmorph | 100       | 32         | gmi:1        | gl2d:1       | last   | norm     |
| transmorph | 100       | 32         | gmi:1        | gl2d:1       | mean   | norm     |
| transmorph | 100       | 32         | gmi:1,ssim:1 | gl2d:1       | last   | norm     |
| transmorph | 100       | 32         | ssim:1       | gl2d:1       | last   | norm     |