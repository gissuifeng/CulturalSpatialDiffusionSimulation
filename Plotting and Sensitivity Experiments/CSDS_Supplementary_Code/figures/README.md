# Figure Workflows

main.py is the entry point. notebooks/ contains one notebook per figure and saved
graphical outputs. environment/requirements.txt lists execution dependencies.
Supply the inputs specified in each notebook before execution. Saved figures can
be viewed directly without running the notebooks.

```shell
python main.py list
python -m pip install -r environment/requirements.txt
python main.py run --figure figure16
```

Executed notebooks are saved to results/. Run python main.py --help for options.
