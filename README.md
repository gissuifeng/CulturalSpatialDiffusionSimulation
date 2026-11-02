# Cultural Spatiotemporal Diffusion Simulation (CSDS)

CSDS simulates inter-city population mobility and within-city cultural diffusion
and acculturation. Cultural populations are represented by counts in city-level
compartments. Two migration sampling modes are available: multinomial and roulette.

## Installation

Use Python 3.11 with the versions listed in requirements.txt.

```shell
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before installing packages and running the model.

## Input preparation

The repository contains model code and documentation. Research datasets are
excluded. Prepare your own synthetic inputs using the four CSV schemas in
[DATA_AND_PREPROCESSING.md](DATA_AND_PREPROCESSING.md), then place them in
`data/source_data/` or set `paths.data_in_dir` in config.yaml.

```shell
python validate_inputs.py
```

The validator checks data/source_data/ and reports structural inconsistencies.
Use `python validate_inputs.py --data path/to/inputs` for another input directory.
Input validation is read-only. Resolve reported errors before simulation.

## Running a simulation

Set the Python hash seed before launching a multinomial run to stabilize cultural
category ordering. In PowerShell:

```powershell
$env:PYTHONHASHSEED="0"
python main.py --mode multinomial --seed 20261018
```

On Linux or macOS:

```shell
PYTHONHASHSEED=0 python main.py --mode multinomial --seed 20261018
```

Use `--mode roulette` for individual migration draws. Use
`--config path/to/config.yaml` for another configuration. Input and output paths
in the configuration are resolved relative to the model directory. The default
configuration specifies 10 annual updates; set `simulation.iterations` to the
required duration. Reusing an output directory overwrites named output files.

## Repository contents

- `main.py`: command-line execution, seeding, and run metadata.
- `core/city.py`: city states and migration sampling probabilities.
- `core/model_multinomial.py`: multinomial migration sampling.
- `core/model_roulette.py`: per-migrant roulette sampling.
- `validate_inputs.py`: input schema and consistency checks.
- `config.yaml`: paths, transition coefficients, and duration.
- `DATA_AND_PREPROCESSING.md`: input construction and preprocessing requirements.
- `MODEL_IMPLEMENTATION.md`: parameters, update sequence, and output interpretation.

## Outputs

`country_output.csv` records city states after each completed update.
`country_output_popflow.csv` records directed migration by cultural state.
`run_metadata.json` records the configuration, sampling mode, seeds, package
versions, and input/code hashes. Outputs are written to `paths.data_out_dir`.
The initial state is supplied by the input tables; iteration-zero rows are not
emitted automatically. Retain the initial inputs when comparing start and end states.

The model uses a fixed annual OD table. Its trajectories describe scenarios
under the supplied mobility and acculturation assumptions. See the implementation
document for the numerical rules and indicator definitions.
