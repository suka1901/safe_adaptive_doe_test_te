# Results

10 seeds, offline emulation on a 41x41 full-factorial dataset (1681 points, 76% safe). Confidence beta = 2.5.

## Experiment 1 - model calibration

| Strategy | Measurements to reach RMSE <= 0.5 % (median) | RMSE at 40 | RMSE at 100 | Trips at 40 | Trips at 100 | Safe region certified at 100 |
|---|---|---|---|---|---|---|
| Uniform grid | 50 (10/10 runs) | 0.57 | 0.29 | 11.0 | 28.0 | 96% |
| Random sampling | 47 (10/10 runs) | 0.82 | 0.32 | 10.3 | 25.5 | 94% |
| Safe active learning (GP) | 42 (10/10 runs) | 0.77 | 0.24 | 0.0 | 0.0 | 97% |

## Experiment 2 - safe Bayesian optimisation (find best safe efficiency)

| Strategy | Measurements to get within 0.5 % of best safe point (median) | Regret at 20 | Regret at 60 | Trips at 60 |
|---|---|---|---|---|
| Uniform grid | not reached | 1.41 | 0.96 | 14.0 |
| Random sampling | not reached | 6.26 | 3.09 | 15.8 |
| Standard Bayesian optimisation (no safety) | 24 (1/10 runs) | 9.28 | 5.67 | 46.1 |
| Safe Bayesian optimisation (SafeOpt-style) | 30 (8/10 runs) | 7.19 | 0.76 | 0.1 |
