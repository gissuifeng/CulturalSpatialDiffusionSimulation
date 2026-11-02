# Spatial Association

## Layout

- main.py: public entry point.
- src/: implementation modules.
- environment/requirements.txt: dependencies.
- data/inputs/: user-supplied inputs.
- results/: generated outputs.

## Execution

```shell
python -m pip install -r environment/requirements.txt
python main.py moran
python main.py blocks
python main.py plot
python main.py all
```

Supply data/inputs/baseline_city_culture_endpoints.csv.gz and POINT shapefiles under data/inputs/geography/city_point_shp/city.shp and province_point_shp/province_point.shp, with DBF/SHX companions and a common metric projected coordinate system. moran computes residual statistics using 4, 8 and 12 neighbors. blocks performs 300 resamples for 250, 500 and 750 km blocks and an offset 500 km grid. plot combines outputs. Results are under results/ and results/block_refit/.

Use --data and --output for alternate directories. Run python main.py --help for all options.
