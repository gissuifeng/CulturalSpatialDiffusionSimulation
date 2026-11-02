# Mobility Intensity

## Layout

- main.py: public entry point.
- src/: implementation modules.
- environment/requirements.txt: dependencies.
- data/inputs/: user-supplied inputs.
- results/: generated outputs.
- config/design.json: experiment settings.

Provide city_pop.csv, city_culture_pop.csv, city_pop_flow.csv and cultural_strength.csv under data/inputs/. The model repository DATA_AND_PREPROCESSING.md defines their schemas.

## Execution

```shell
python -m pip install -r environment/requirements.txt
python main.py test
python main.py run --workers 2
python main.py analyze
python main.py all --workers 2
```

Annual origin outflows are scaled by 0.8, 1.0 and 1.2 with ten seeds per scenario. run writes per-run states and metrics. analyze writes paired differences and endpoint summaries to results/analysis/. all executes both stages.

Use --data and --output for alternate directories. Run python main.py --help for all options.
Use --config to select another experiment configuration.
