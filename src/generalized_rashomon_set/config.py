import os
from pathlib import Path
import logging
import time
Path("../logs").mkdir(parents=True, exist_ok=True)
Path("../results").mkdir(parents=True, exist_ok=True)
ROOT_DIR = os.getcwd()
OUTPUT_DIR = ROOT_DIR + '/../results'
LOG_DIR = ROOT_DIR + '/../logs'
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.FileHandler(LOG_DIR + "/log_{}.log".format(time.strftime("%Y%m%d-%H%M%S")),
                                                  mode='w'),
                              stream_handler])
logger = logging.getLogger(__name__)
logger.info('The output directory is {}\n The log directory is {}'.format(OUTPUT_DIR, LOG_DIR))
time_str = time.strftime("%Y%m%d-%H%M%S")