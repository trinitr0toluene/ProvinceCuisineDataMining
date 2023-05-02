import os
import logging
import pickle
import torch
import torch.distributed as dist


class Log():
    def __init__(self):
        self.logger = None
        
    def get_logger(self):
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.INFO)
        

    def add_handler(self, time):
        # path = os.path.join("Log", time)
        path = os.path.join("Log", time[:10], time[11:])
        if os.path.exists(path) == False:
            os.makedirs(path)

        log_path = os.path.join(path, "record.log")
        fh = logging.FileHandler(log_path, mode='w')
        formatter = logging.Formatter("%(levelname)s - %(process)d - %(asctime)s - %(filename)s: %(message)s")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

  