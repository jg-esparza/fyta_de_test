from pathlib import Path

import hydra
from omegaconf import DictConfig

from .logging_utils import configure_logging
from .io import ensure_dir, load_inputs


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(str(cfg.logging.level))
    data = load_inputs(Path(str(cfg.data_dir)), dict(cfg.files))



if __name__ == "__main__":
    main()
