"""Shared loaders for the post-hoc R1 analyses: the official scorer, the R1 gold file, the frozen R1 probe predictions
and the arm file map / paired Poisson bootstrap of robustness/r1/annotation/rescore_after_review.py."""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scorer(path=None):
    return load_module('multitq_eval', path or os.path.join(REPO, 'src', 'multitq_eval.py'))


def rescorer():
    return load_module('rescore_after_review', os.path.join(REPO, 'robustness', 'r1', 'annotation', 'rescore_after_review.py'))


def gold(path):
    return {int(q): v for q, v in json.load(open(path))['questions'].items()}
