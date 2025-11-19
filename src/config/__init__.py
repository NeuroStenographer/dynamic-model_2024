from omegaconf import OmegaConf
from pathlib import Path

Config = OmegaConf.create()
# read all the yaml files in src.config
config_path = Path(__file__).parent
for yaml_file in config_path.glob('**/*.yaml'):
    new_conf = OmegaConf.load(yaml_file)
    try:
        Config = OmegaConf.merge(Config, new_conf)
    except Exception as e:
        message = f'Error merging {yaml_file} into Config: {e}'
        raise Exception(message)


