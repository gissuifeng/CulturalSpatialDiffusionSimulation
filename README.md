# CSDS Code

This package contains the model implementation and supplementary analysis and visualization code for the Cultural Spatiotemporal Diffusion Simulation (CSDS) study.

## CSDS

The `CSDS/` folder contains the CSDS model implementation, including inter-city population mobility, within-city cultural diffusion and acculturation, model configuration, and input validation. Its documentation describes the input schemas, numerical implementation, and simulation outputs.

## Plotting and Sensitivity Experiments

The `Plotting and Sensitivity Experiments/CSDS_Supplementary_Code/` folder contains the following supplementary experimental and visualization workflows:

| Subfolder | Purpose |
|---|---|
| `parameter_sensitivity/` | Factorial sensitivity analysis of cultural-transition parameters. |
| `mobility_intensity/` | Experiments examining the effects of different population-mobility intensities. |
| `spatial_association/` | Residual spatial autocorrelation analysis and spatial-block resampling of distance-based diffusion curves. |
| `figures/` | Figure-specific Jupyter notebooks with saved graphical outputs. |

Each workflow includes a `main.py` entry point and a README describing its configuration, required inputs, execution, and outputs.
