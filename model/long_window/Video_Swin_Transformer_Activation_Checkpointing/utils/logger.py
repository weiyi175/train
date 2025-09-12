#!/usr/bin/env python3
import logging, sys, os, json
from pathlib import Path


def get_logger(name: str, log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        fh = logging.FileHandler(Path(log_dir)/'train.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

def append_jsonl(path: str, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False)+'\n')

__all__ = ['get_logger', 'append_jsonl']
