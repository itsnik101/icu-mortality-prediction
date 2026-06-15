import os
import random
import logging
import numpy as np
import joblib
from datetime import datetime
from typing import Any
import config 

def setup_logger(module_name:str) -> logging.Logger:
    logger=logging.getLogger(module_name)
    logger.setLevel(logging.info)
    
    if not logger.handlers:
        
        formatter=logger.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    log_dir=config.ROOT_DIR / "logs"
    log_dir.mkdir(parents=True,exist_ok=True)
    
    central_log_file=log_dir / "Pipeline_exceution.log"
    file_handler=logging.FileHandler(central_log_file,mode='a',encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

utils_logger=setup_logger("utils_infra")

def enforce_reproducibility(seed:int | None = None) -> None:
    target_seed=seed if seed is not None else config.SEED
    random.seed(target_seed)
    os.environ['PYTHONHASHSEED']= str(target_seed)
    np.random.seed(target_seed)
    
    utils_logger.info(f"Global pseudorandom states locked securely using seed footprint: {target_seed}")
    utils_logger.warning("Reminder: Estimator-level routines require internal 'random_state' parameter assignment.")

def save_artifact(artifact: Any , filename : str) -> None:
    try:
        
        config.MODEL_DIR.mkdir(parents=True,exist_ok=True)
        target_path=config.MODEL_DIR / filename

        joblib.dump(artifact,target_path,compress=3)
        utils_logger.info(f"Successfully serialized production asset artifact to disk destination: {target_path}")
    except Exception as e:
        utils_logger.error(f"De-serialization mapping sequence failed for target resource {filename}: {str(e)}")  
        raise RuntimeError(f"Corrupt binary framework allocation block detected: {str(e)}") 