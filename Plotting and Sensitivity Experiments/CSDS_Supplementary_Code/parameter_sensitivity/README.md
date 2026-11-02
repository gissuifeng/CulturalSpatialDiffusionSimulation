# Parameter Sensitivity

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
python main.py plan
python main.py smoke --output results/smoke
python main.py run --workers 2
python main.py verify
python main.py summarize
python main.py analyze
```

The configuration specifies five levels of r, gamma and theta with three seeds per combination (375 factorial runs), plus control runs. analyze reads the summary tables produced by summarize. D_TV_pp measures total variation in normalized cultural shares; state percentages use model population.

Use --data and --output for alternate directories. Run python main.py --help for all options.
Use --config to select another experiment configuration.
