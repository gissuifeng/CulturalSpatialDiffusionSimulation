# CSDS Supplementary Analysis Code

Each workflow has one main.py entry point and one README.md. Implementation modules
are under src/, settings under config/, and dependency lists under environment/.
Users provide research inputs under data/inputs/. Generated outputs use results/.

| Directory | Workflow |
|---|---|
| parameter_sensitivity | Factorial parameter sensitivity |
| mobility_intensity | Three mobility scenarios with ten replicates |
| spatial_association | Residual Moran analysis and spatial-block resampling |
| figures | Figure notebooks with saved graphical outputs |

Run python main.py --help in a workflow directory for its available commands.
Research datasets and numerical result files are excluded.
