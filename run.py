import warnings
from pathlib import Path

from omegaconf import OmegaConf
from dask.distributed import Client

# filter some warnings
# warnings.filterwarnings("ignore", ".*does not have many workers.*")
# warnings.filterwarnings("ignore", ".*Detected KeyboardInterrupt, attempting graceful shutdown....*")

# python run.py name="per_epoch" nb_workers=1 repeat_tuning=1 \
#         log_base_dir="logs" progress_bar=False \
#         save_train_metrics=True save_val_metrics=True remove_checkpoints=False \
#         selected_dataset_groups=["uci"] \
#         tuning_type="QRT_per_epoch"

def main():
    import sys

    from uq.configs.config import get_config
    from uq import utils
    from uq.runner import run_all

    config = OmegaConf.from_cli(sys.argv)

    for dataset in ['kin8nm', 'boston', 'yacht', 'wine', 'concrete', 'energy', 'mpg', 'power', 'naval', 'protein']: # 'boston', 'yacht', 'wine', 'concrete', 'energy', 'naval', 'protein', 'mpg', 'power', 'kin8nm'
        for seed_id in range(1, 6):
            config = OmegaConf.from_cli(sys.argv)
            print(seed_id)
            config.seed = seed_id
            config.name = f'{dataset}_{seed_id}'
            config = get_config(config)
            config.device = 'cuda'
            # config.clean_previous = True
            OmegaConf.resolve(config)
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            print(config.log_dir)
            # Pretty print config using Rich library
            # if config.get("print_config"):
            #     utils.print_config(config, resolve=True)
            # Set parallelization
            manager = 'joblib'
            if config.nb_workers == 1:
                manager = 'sequential'
            if manager == 'dask':
                Client(n_workers=config.nb_workers, threads_per_worker=1, memory_limit=None)
            # Train model
            run_all(config, manager=manager)
    return


if __name__ == "__main__":
    main()
