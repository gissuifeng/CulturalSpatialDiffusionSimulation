"""
main.py
Execution entry point for the Cultural Spatiotemporal Diffusion Simulation (CSDS).
"""
import logging
import os
import yaml
import argparse
import random
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd

# Configure top-level logging for the project
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - CSDS Model - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. Set up argument parsing for simulation modes
    parser = argparse.ArgumentParser(description="Cultural Spatiotemporal Diffusion Simulation (CSDS)")
    parser.add_argument(
        '--mode', 
        type=str, 
        default='multinomial', 
        choices=['multinomial', 'roulette'],
        help='Select simulation mode: "multinomial" (default, fast matrix sampling) or "roulette".'
    )
    parser.add_argument('--config', default=str(Path(__file__).parent / 'config.yaml'))
    parser.add_argument('--seed', type=int, default=20261018,
                        help='Seed for Python and NumPy random generators.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == 'multinomial' and os.environ.get('PYTHONHASHSEED') is None:
        logging.warning('Set PYTHONHASHSEED before launch for stable cultural category ordering.')

    logging.info(f"Initializing CSDS Project in [{args.mode.upper()}] mode...")
    
    try:
        # 2. Load configuration from YAML
        config = load_config(args.config)
        
        # 3. Dynamically import the correct model based on the selected mode
        if args.mode == 'multinomial':
            from core.model_multinomial import CSDSModel
        elif args.mode == 'roulette':
            from core.model_roulette import CSDSModel
            
        # 4. Initialize the core model
        csds_model = CSDSModel(config)
        
        # 5. Pipeline Execution
        csds_model.load_data()
        csds_model.run()
        csds_model.export_results()
        project = Path(__file__).resolve().parent
        hashed = list(Path(csds_model.data_dir).glob('*.csv')) + list((project / 'core').glob('*.py')) + [project / 'main.py']
        metadata = {
            'mode': args.mode, 'seed': args.seed,
            'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
            'config': config, 'python': platform.python_version(),
            'numpy': np.__version__, 'pandas': pd.__version__,
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in hashed}
        }
        (Path(csds_model.output_dir) / 'run_metadata.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8')
        
    except FileNotFoundError as e:
        logging.error(f"Missing required file: {e}")
        logging.info("Please ensure config and data directories are correctly set up.")
        raise SystemExit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during simulation: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()