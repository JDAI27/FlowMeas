#!/usr/bin/env python3
"""
Run experiments from JSON configuration file.
Supports single experiments, batch experiments, and parameter sweeps.
"""

import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add the directory containing main.py to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ExperimentConfig, run_experiment


def load_json_config(json_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def expand_parameter_sweeps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand parameter sweeps into individual configurations.
    
    If a parameter value is a dict with "sweep" key, expand it.
    Example: {"beta": {"sweep": [1e4, 2e4, 5e4]}} -> 3 configs
    """
    sweep_params = {}
    base_config = {}
    
    # Identify sweep parameters
    for key, value in config.items():
        if isinstance(value, dict) and "sweep" in value:
            sweep_params[key] = value["sweep"]
        else:
            base_config[key] = value
    
    if not sweep_params:
        return [config]
    
    # Generate all combinations
    configs = []
    
    # Simple implementation for single parameter sweeps
    if len(sweep_params) == 1:
        param_name, values = list(sweep_params.items())[0]
        for value in values:
            new_config = base_config.copy()
            new_config[param_name] = value
            configs.append(new_config)
    else:
        # For multiple parameter sweeps, use itertools.product
        import itertools
        param_names = list(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]
        
        for values in itertools.product(*param_values):
            new_config = base_config.copy()
            for name, value in zip(param_names, values):
                new_config[name] = value
            configs.append(new_config)
    
    return configs


def create_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Convert dictionary to ExperimentConfig."""
    # Handle nested dictionaries for reward_kwargs and objective_kwargs
    if "reward_kwargs" in config_dict and isinstance(config_dict["reward_kwargs"], dict):
        config_dict["reward_kwargs"] = config_dict["reward_kwargs"]
    else:
        config_dict["reward_kwargs"] = {}
    
    if "objective_kwargs" in config_dict and isinstance(config_dict["objective_kwargs"], dict):
        config_dict["objective_kwargs"] = config_dict["objective_kwargs"]
    else:
        config_dict["objective_kwargs"] = {}
    
    return ExperimentConfig(**config_dict)


async def run_single_experiment(config_dict: Dict[str, Any], exp_name: str = None):
    """Run a single experiment with the given configuration."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Running experiment: {exp_name or 'Unnamed'}")
    logging.info(f"{'='*60}")
    
    # Create ExperimentConfig
    exp_config = create_experiment_config(config_dict)
    
    # Print key parameters
    logging.info(f"Hamiltonian: {exp_config.hamiltonian_path}")
    logging.info(f"Beta: {exp_config.beta}")
    logging.info(f"Epsilon: {exp_config.epsilon}")
    logging.info(f"N measurements: {exp_config.n_measurements}")
    logging.info(f"Hidden dim: {exp_config.hidden_dim}")
    logging.info(f"Learning rate: {exp_config.lr}")
    
    # Run experiment
    results, trainer = await run_experiment(exp_config)
    
    return results, trainer


async def run_all_experiments(json_path: str):
    """Load and run all experiments from JSON file."""
    config = load_json_config(json_path)
    
    # Check if it's a single experiment or multiple
    if "experiments" in config:
        # Multiple experiments
        experiments = config["experiments"]
        
        # Check for common parameters
        common_params = config.get("common", {})
        
        for exp_name, exp_config in experiments.items():
            # Merge common parameters
            full_config = {**common_params, **exp_config}
            
            # Handle parameter sweeps
            expanded_configs = expand_parameter_sweeps(full_config)
            
            for i, config_instance in enumerate(expanded_configs):
                if len(expanded_configs) > 1:
                    name = f"{exp_name}_sweep_{i+1}"
                else:
                    name = exp_name
                
                await run_single_experiment(config_instance, name)
    else:
        # Single experiment
        expanded_configs = expand_parameter_sweeps(config)
        
        for i, config_instance in enumerate(expanded_configs):
            name = f"experiment_{i+1}" if len(expanded_configs) > 1 else "experiment"
            await run_single_experiment(config_instance, name)


def main():
    parser = argparse.ArgumentParser(description="Run GFlowNet experiments from JSON config")
    parser.add_argument("config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without running")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logging.error(f"Configuration file '{args.config}' not found")
        sys.exit(1)
    
    if args.dry_run:
        config = load_json_config(args.config)
        logging.info("Loaded configuration:")
        logging.info(json.dumps(config, indent=2))
    else:
        asyncio.run(run_all_experiments(args.config))


if __name__ == "__main__":
    main()
