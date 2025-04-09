import logging
import colorlog
import sys
import torch

from collections import defaultdict

from .shared_utils import SmoothedValue


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        keys = sorted(self.meters)
        # for name, meter in self.meters.items():
        for name in keys:
            meter = self.meters[name]
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


def make_logger(name, level=logging.INFO, local_rank=0, filepath=None):
    """
    Attach a stream handler to all loggers.

    Parameters
    ------------
    name : str
        Name of the logger
    level : enum (int)
        Logging level, like logging.INFO
    capture_warnings: bool
        If True capture warnings
    filepath: None or str
        path to save the logfile

    Returns
    -------
    logger: Logger object
        Logger attached with a stream handler
    """

    # make sure we log warnings from the warnings module
    # logging.captureWarnings(capture_warnings)

    if local_rank is not None:
        name = f"{name} rank {local_rank}"

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(name)s] [%(asctime)s] [(%(filename)s:%(lineno)3s)] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            # 'INFO': 'green',
            'DEBUG': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        },
        secondary_log_colors={},
        style='%'
    )

    # create a basic formatter
    # formatter = logging.Formatter(formatter)

    # if no handler was passed use a StreamHandler
    logger = logging.getLogger(name)
    
    # if in multiprocessing there a different processed and the logging level is set to INFO, do not print for every process but only for main.
    # however, do print if there is a warning or if in debug mode
    if level == logging.INFO and local_rank != 0:
        level = logging.WARNING
        
    logger.setLevel(level)
    logger.propagate = 0

    if not any([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]):
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    formatter = logging.Formatter("[%(name)s] [%(asctime)s] [(%(filename)s:%(lineno)3s)] [%(levelname)s] %(message)s")

    if filepath and not any([isinstance(handler, logging.FileHandler) for handler in logger.handlers]):
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # set nicer numpy print options
    # np.set_printoptions(precision=3, suppress=True)

    # logger.addHandler(logging.StreamHandler())

    return logger
