#!/usr/bin/env python3
"""
Run GFlowNet experiments from JSON configuration. Handles cluster preemption signals.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path to import from code module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main.py
from code.main import ExperimentConfig, run_experiment

# Global flag for graceful shutdown
_shutdown_requested = False


def setup_logging(log_file: Optional[str] = None):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )


def validate_config(config_dict: Dict[str, Any]) -> None:
    required_fields = ['hamiltonian_path']
    
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Missing required field: {field}")
    
    # Check if hamiltonian file exists
    hamiltonian_path = config_dict['hamiltonian_path']
    if not Path(hamiltonian_path).exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        full_path = project_root / hamiltonian_path
        if full_path.exists():
            config_dict['hamiltonian_path'] = str(full_path)
        else:
            raise FileNotFoundError(f"Hamiltonian file not found: {hamiltonian_path}")


def load_config_from_json(json_path: str) -> ExperimentConfig:
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    # Validate configuration
    validate_config(config_dict)
    
    # Set default model_type if not specified
    if 'model_type' not in config_dict:
        config_dict['model_type'] = 'clifford_mlp'
    
    if 'async_eval' not in config_dict:
        config_dict['async_eval'] = True
    
    config = ExperimentConfig(**config_dict)
    
    return config


def setup_mila_environment():
    """Setup environment variables for Mila cluster."""
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_cpus = os.environ['SLURM_CPUS_PER_TASK']
        os.environ['OMP_NUM_THREADS'] = n_cpus
        os.environ['MKL_NUM_THREADS'] = n_cpus
        logging.info(f"Set thread counts to {n_cpus} based on SLURM allocation")
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    if 'SLURM_JOB_ID' in os.environ:
        logging.info(f"Running on Mila cluster - Job ID: {os.environ['SLURM_JOB_ID']}")
        if 'SLURM_NODELIST' in os.environ:
            logging.info(f"Node: {os.environ['SLURM_NODELIST']}")
        
        restart_count = os.environ.get('SLURM_RESTART_COUNT', '0')
        if int(restart_count) > 0:
            logging.info(f"This is a REQUEUED job (restart #{restart_count})")


def handle_preemption_signal(signum, frame):
    """Handle preemption signals for graceful shutdown with checkpoint save."""
    global _shutdown_requested
    
    signal_name = signal.Signals(signum).name
    logging.warning(f"\n{'='*60}")
    logging.warning(f"Received {signal_name} signal - Preemption/timeout imminent!")
    logging.warning(f"Time: {datetime.now()}")
    logging.warning(f"{'='*60}")
    
    _shutdown_requested = True
    
    try:
        from code.main import _current_trainer, _current_results_dir, convert_metrics_history_to_cpu
        
        if _current_trainer is not None and _current_results_dir is not None:
            logging.info("Attempting to save checkpoint before shutdown...")
            checkpoint_path = _current_results_dir / 'checkpoint_update.pth'
            
            current_update = 0
            if hasattr(_current_trainer, 'metrics_history') and _current_trainer.metrics_history:
                current_update = len(_current_trainer.metrics_history.get('loss', []))
            
            metrics_history_cpu = convert_metrics_history_to_cpu(_current_trainer.metrics_history)
            _current_trainer.gfn.save_checkpoint(str(checkpoint_path), current_update, metrics_history_cpu)
            
            logging.info(f"Emergency checkpoint saved at update {current_update}")
            logging.info(f"Checkpoint path: {checkpoint_path}")
        else:
            logging.warning("Trainer not yet initialized - no checkpoint to save")
    except Exception as e:
        logging.error(f"Failed to save emergency checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("Exiting gracefully. Job will be requeued if --requeue is set.")
    sys.exit(99)


def setup_signal_handlers():
    """Setup signal handlers for graceful preemption handling."""
    signal.signal(signal.SIGTERM, handle_preemption_signal)
    signal.signal(signal.SIGUSR1, handle_preemption_signal)
    
    logging.info("Signal handlers installed for graceful preemption handling")


async def main():
    parser = argparse.ArgumentParser(
        description="Run GFlowNet experiment from JSON configuration (Mila cluster compatible)"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (in addition to stdout)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running experiment"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Setup signal handlers for graceful preemption
    setup_signal_handlers()
    
    # Setup Mila environment if on cluster
    setup_mila_environment()
    
    # Log start time
    start_time = datetime.now()
    logging.info("=" * 60)
    logging.info("GFlowNet Experiment Runner")
    logging.info("=" * 60)
    logging.info(f"Start time: {start_time}")
    logging.info(f"Configuration file: {args.config}")
    
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        config = load_config_from_json(args.config)
        logging.info("Configuration loaded successfully")
        
        restart_count = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
        if restart_count > 0 and not config.resume:
            logging.info(f"\n{'='*60}")
            logging.info(f"REQUEUED JOB DETECTED (restart #{restart_count})")
            logging.info(f"Forcing resume=True to continue existing experiment")
            logging.info(f"{'='*60}\n")
            config.resume = True
        
        logging.info("\nExperiment Parameters:")
        logging.info(f"  Hamiltonian: {config.hamiltonian_path}")
        logging.info(f"  Results directory: {config.results_dir}")
        logging.info(f"  Device preference: {config.device_preference}")
        logging.info(f"  Number of updates: {config.n_updates}")
        logging.info(f"  Measurements per batch: {config.n_measurements}")
        logging.info(f"  Max circuit depth: {config.max_depth}")
        logging.info(f"  Beta (temperature): {config.beta}")
        logging.info(f"  Learning rate: {config.lr}")
        logging.info(f"  Hidden dimension: {config.hidden_dim}")
        logging.info(f"  Hidden layers: {config.num_hidden_layers}")
        logging.info(f"  Reward type: {config.reward_type}")
        logging.info(f"  Objective: {config.objective_type}")
        logging.info(f"  Resume from checkpoint: {config.resume}")
        logging.info(f"  Async evaluation: {config.async_eval}")
        
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if args.validate_only:
        logging.info("\nConfiguration validation successful!")
        sys.exit(0)
    
    logging.info("\nStarting experiment...")
    try:
        results, trainer = await run_experiment(config)
        
        if _shutdown_requested:
            logging.info("\nShutdown was requested during training.")
            logging.info("Checkpoint should have been saved.")
            sys.exit(99)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info("\n" + "=" * 60)
        logging.info("Experiment completed successfully!")
        logging.info(f"End time: {end_time}")
        logging.info(f"Total duration: {duration}")
        logging.info("=" * 60)
        
    except KeyboardInterrupt:
        logging.info("\nExperiment interrupted by user")
        try:
            from code.main import _current_trainer, _current_results_dir, convert_metrics_history_to_cpu
            if _current_trainer is not None and _current_results_dir is not None:
                checkpoint_path = _current_results_dir / 'checkpoint_update.pth'
                current_update = len(_current_trainer.metrics_history.get('loss', [])) if hasattr(_current_trainer, 'metrics_history') else 0
                metrics_history_cpu = convert_metrics_history_to_cpu(_current_trainer.metrics_history)
                _current_trainer.gfn.save_checkpoint(str(checkpoint_path), current_update, metrics_history_cpu)
                logging.info(f"Checkpoint saved at update {current_update} before exit")
        except Exception as e:
            logging.error(f"Failed to save checkpoint on interrupt: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())