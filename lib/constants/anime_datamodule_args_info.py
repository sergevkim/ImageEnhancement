from pathlib import Path


ANIME_DATAMODULE_ARGS_INFO = {
    'batch_size': {
        'default': 10,
        'type': int,
    },
    'data_dir': {
        'default': f'{Path.cwd()}/data',
        'type': str,
    },
    'num_workers': {
        'default': 1,
        'type': int,
    },
    'test_size': {
        'default': 1000,
        'type': int,
    },
    'train_size': {
        'default': 7000,
        'type': int,
    },
    'val_size': {
        'default': 2000,
        'type': int,
    },
}

