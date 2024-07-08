import numpy as np
from utils import *

NAME_TO_FIELD = { # maybe change 
    'OpenWebText2': 'owt2',
    'Base': 'base',
    'Tuned': 'tuned',
    'Seed': 'seed',
    'Sweep': 'sweep',
    'RefinedWeb': 'rw',
    'Short': 'short',
    'Long': 'long',
    'Const': 'const',
    'Kaplan': 'kaplan',
    'Chinchilla': 'chinchilla',
}


FLOP_VALS = 1e17 * 2 ** np.arange(-3.0, 9, 1.0)

CHINCHILLA_FLOPS = 5.88e23
CHINCHILLA_STR = 'C_{C}'

PARAM_COUNT_TO_FLOP_TOK_KEY = {
    'standard': 'flops_per_token',
    'attention': 'flops_per_token_att',
    'kaplan': 'flops_per_token_no_att_no_embed',
}


# TODO refactoring this into something suitable for working with one cleaned DF
# things like the arg dict etc should be part of the analysis / data reading code and not in notebooks
CONFIG_DICT_LABEL = {
    ('rw', 'base', 'long', 'kaplan', 'kaplan', 'train'): 'Reproducing Kaplan et al.',
    ('rw', 'base', 'long', 'kaplan', 'standard', 'val'): 'Counting last layer FLOPs',
    ('rw', 'base', 'short', 'kaplan', 'standard', 'val'): 'Correcting warmup',
    ('rw', 'base', 'short', 'chinchilla', 'standard', 'val'): 'Cosine decay',
    ('rw', 'tuned', 'short', 'const', 'standard', 'val'): 'Optimizer tuning (no decay)',
    ('rw', 'tuned', 'short', 'const', 'standard', 'train'): 'Optimizer tuning (no decay) - train',
    ('rw', 'tuned', 'long', 'const', 'kaplan', 'train'): '', # kaplan tuned
    ('rw', 'base', 'long', 'kaplan', 'attention', 'train'): 'Counting last layer\nand attention FLOPs',
    ('rw', 'base', 'short', 'kaplan', 'attention', 'train'): 'Correcting warmup',
    ('rw', 'base', 'short', 'chinchilla', 'attention', 'train'): 'Cosine decay',
    ('rw', 'tuned', 'short', 'const', 'attention', 'train'): 'Optimizer tuning (no decay)',
}

CONFIG_DICT_COLOR = {
    ('rw', 'base', 'long', 'kaplan', 'kaplan', 'train'): get_color(4),
    ('rw', 'base', 'long', 'kaplan', 'standard', 'val'): get_color(0),
    ('rw', 'base', 'short', 'kaplan', 'standard', 'val'): get_color(1),
    ('rw', 'base', 'short', 'chinchilla', 'standard', 'val'): get_color(2),
    ('rw', 'tuned', 'short', 'const', 'standard', 'val'): get_color(3),

    ('rw', 'tuned', 'long', 'const', 'kaplan', 'train'): get_color(6),
    ('rw', 'sweep', 'short', 'const', 'standard', 'train'): get_color(7),

    ('rw', 'base', 'long', 'kaplan', 'attention', 'train'): get_color(0),
    ('rw', 'base', 'short', 'kaplan', 'attention', 'train'): get_color(1),
    ('rw', 'base', 'short', 'chinchilla', 'attention', 'train'): get_color(2),
    ('rw', 'tuned', 'short', 'const', 'attention', 'train'): get_color(3),

}

CONFIG_DICT_MARKER = {
    ('rw', 'base', 'long', 'kaplan', 'kaplan', 'train'): get_marker(4),
    ('rw', 'base', 'long', 'kaplan', 'standard', 'val'): get_marker(0),
    ('rw', 'base', 'short', 'kaplan', 'standard', 'val'): get_marker(1),
    ('rw', 'base', 'short', 'chinchilla', 'standard', 'val'): get_marker(2),
    ('rw', 'tuned', 'short', 'const', 'standard', 'val'): get_marker(3),

    ('rw', 'tuned', 'long', 'const', 'kaplan', 'train'): get_marker(6),

    ('rw', 'base', 'long', 'kaplan', 'attention', 'train'): get_marker(0),
    ('rw', 'base', 'short', 'kaplan', 'attention', 'train'): get_marker(1),
    ('rw', 'base', 'short', 'chinchilla', 'attention', 'train'): get_marker(2),
    ('rw', 'tuned', 'short', 'const', 'attention', 'train'): get_marker(3),
}

for d in (CONFIG_DICT_LABEL, CONFIG_DICT_COLOR, CONFIG_DICT_MARKER):
    d_new = {}
    for (a,b,c,d_,e,f),v in d.items():
        d_new['owt2', b, c, d_, e, f] = v
    d.update(d_new)

FIGURE1_CONFIGS = [
    ('rw', 'base', 'long', 'kaplan', 'kaplan', 'train'),
    ('rw', 'base', 'long', 'kaplan', 'standard', 'val'),
    ('rw', 'base', 'short', 'kaplan', 'standard', 'val'),
    ('rw', 'base', 'short', 'chinchilla', 'standard', 'val'),
    ('rw', 'tuned', 'short', 'const', 'standard', 'val')
]

FIGURE1_CONFIGS_OWT2 = [("owt2", b, c, d, e, f) for _, b, c, d, e, f in FIGURE1_CONFIGS]

WARMUP_EVIDENCE_CONFIGS = [
    ('rw', 'base', 'long', 'kaplan', 'standard', 'val'),
    ('rw', 'base', 'short', 'kaplan', 'standard', 'val'),
]

ATTENTION_ACCOUNTING_CONFIGS = [
            ('rw', 'base', 'long', 'kaplan', 'attention', 'train'),
            ('rw', 'base', 'short', 'kaplan', 'attention', 'train'),
            ('rw', 'base', 'short', 'chinchilla', 'attention', 'train'),
            ('rw', 'tuned', 'short', 'const', 'attention', 'train'),
            ]

SWEEP_CONFIGS = [('rw', 'sweep', 'short', 'const', 'standard', 'train')]

TUNED_TRAIN_CONFIG = [('rw', 'tuned', 'short', 'const', 'standard', 'train')]

KEYS_TO_TITLE_SWEEP = {'bs': 'Batch size', 'lr': 'Learning rate'}

ISOFLOP_ARGS = {
    ('kaplan', 'train'): dict(loss_key='train/loss_smoothed', flop_per_token_key='flops_per_token_no_att_no_embed', n_key='params_no_embed'),
    ('standard', 'val'):  dict(loss_key='val/loss', flop_per_token_key='flops_per_token', n_key='params'),
    ('standard', 'train'): dict(loss_key='train/loss_smoothed', flop_per_token_key='flops_per_token', n_key='params'),
    ('attention', 'train'): dict(loss_key='train/loss_smoothed', flop_per_token_key='flops_per_token_att', n_key='eff_params_att'),
}


RW_SEED_CONFIG = dict(noise_low=0.002, noise_high=0.05, l_threshold_high=7, l_threshold_low=3)
OWT2_SEED_CONFIG = dict(noise_low=0.01, noise_high=0.1, l_threshold_high=6, l_threshold_low=3)
SEED_ARGS = {k:RW_SEED_CONFIG for k in CONFIG_DICT_COLOR}

SEED_ARGS.update({('owt2', b, c, d, e, f): OWT2_SEED_CONFIG for _, b, c, d, e, f in CONFIG_DICT_COLOR})

SEED_ARGS.update({('rw', 'tuned', 'short', 'const', 'standard', 'train'): RW_SEED_CONFIG})
SEED_ARGS.update({('rw', 'tuned', 'short', 'const', 'kaplan', 'train'): RW_SEED_CONFIG})