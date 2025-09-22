import logging

logger = None

def set_logger(save_directory):
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(save_directory),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)