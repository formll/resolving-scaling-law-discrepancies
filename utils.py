import numpy as np
import matplotlib
from typing import Sequence
from configs import *

MARKERS = ['o', 'v', 'p', 's','X', '<',  '^','P', '*', '>', 'd']
LINESTYLES = ['-', '--', ':', '-.']
COLORS = [c for c in matplotlib.colors.TABLEAU_COLORS]
COLORS.append('indigo')
COLORS.append('tan')#[:-2]
COLORS.append('k')

def get_marker(i):
    return MARKERS[i % len(MARKERS)]


def get_color(i, clist=COLORS):
    return clist[i % len(clist)]


def get_linestyle(i):
    return LINESTYLES[i % len(LINESTYLES)]

def round_up_to_first_decimal(num):
    if num == 0:
        return 0
    magnitude = 10 ** np.floor(np.log10(abs(num)))
    first_significant_digit = np.ceil(num / magnitude)
    rounded_number = first_significant_digit * magnitude

    return rounded_number


def round_down_to_first_decimal(num):
    if num == 0:
        return 0
    magnitude = 10 ** np.floor(np.log10(abs(num)))
    first_significant_digit = np.floor(num / magnitude)
    rounded_number = first_significant_digit * magnitude

    return rounded_number


def nanfloat(x):
    try:
        return float(x)
    except:
        return np.nan


def nice_string(s):
    return s.replace('_', ' ').capitalize()


def short_string(s, strlen=12):
    return '_'.join([ss[:strlen] for ss in s.split('_')])


def float_fmt(s):
    if isinstance(s, float):
        return f'{s:.4g}'
    else:
        return s

def config_str(ks, vs, key_len=12, sep=', '):
    if not isinstance(vs, (Sequence, np.ndarray)):
        vs = [vs]
    return sep.join([f'{short_string(k, strlen=key_len)} = {float_fmt(v)}' for k, v in zip(ks, vs)])


def fmt_model_size(nn, key='n'):
    for letter, value in dict(M=1e6, B=1e9, T=1e12).items():
        if key != 'multiplier':
            if nn < 1000 * value:
                return f'${np.round(nn/value):g}' + r'\textrm{' + letter + r'}$'
        else:
            small_str = fr'${nn:.2f}$'
            # if there is a non-zero digit, return the string. don't forget decimal point
            if any([d != '0' for d in small_str if d not in ['.', '-', '$']]):
                return small_str
            return fr'${nn:.1g}$'


def maybe_get_item(x, i):
    try:
        return x[i]
    except:
        return x