import numpy as np
import pandas as pd

def precise_flops_per_token_chinchilla(width, depth):
    seq_len = 2048
    vocab_size = 50432
    num_heads = 4
    width = width.astype(float)
    depth = depth.astype(float)

    embeddings = 2 * seq_len * vocab_size * width

    attention = 2 * 3 * seq_len * (width ** 2)
    kq_logits = 2 * seq_len * seq_len * width
    softmax = 3 * num_heads * seq_len * seq_len
    softmax_q_red = 2 * seq_len * seq_len * width
    final_linear = 2 * seq_len * (width ** 2)
    attention += kq_logits + softmax + softmax_q_red + final_linear

    ffw_size = 4 * width # check this, in the paper it is 4 * width
    dense_block = 4 * seq_len * width * ffw_size
    final_logits = 2 * seq_len * width * vocab_size
    forward_pass = embeddings + depth * attention + depth * dense_block + final_logits
    backward_pass = 2 * forward_pass
    return (forward_pass + backward_pass) / seq_len

def precise_param_count_open_lm(width, depth, vocab_size=50432):
    d_ff = 256 * (((2 * 4 * width / 3).astype(int) + 256 - 1) // 256)
    return (4 * width + 3 * d_ff) * width * depth + vocab_size * width

def proportional_sliding_window_filter(x, p=0.05):
    # assert that the index of x has constant increments?
    x_cumsum = x.cumsum().values
    x_cumsum_pad = np.concatenate([[0], x_cumsum])
    inds = np.arange(len(x))
    inds_up = np.minimum(inds + np.floor(p * inds).astype(int), len(x)-1)
    inds_down = np.maximum(0, inds - np.floor(p * inds).astype(int))
    inds_new = (inds_up + inds_down)/2
    index_new = np.interp(inds_new, inds, x.index)
    return pd.Series((x_cumsum[inds_up] - x_cumsum_pad[inds_down]) / (inds_up - inds_down+1), 
                     index=index_new, name=x.name + '_smoothed')

def apply_smoothing_filter(df, filter_func, compensate_for_logging_delay=True, key='train/loss', **filter_args):
    out = []
    for _, row in df.iterrows():
        if len(row[key]) == 0:
            out.append(None)
            continue
        filtered = filter_func(row[key].dropna(), **filter_args)
        if compensate_for_logging_delay:
            filtered.index = filtered.index - np.diff(filtered.index, prepend=0)/2
        out.append(filtered)
    return out

def process_big_df(big_df):
    big_df = big_df.copy()

    # Counting parameters
    big_df['params_active'] = (12 * (big_df.width**2) * big_df.depth + big_df.vocab_size * big_df.width).astype(float)
    big_df['params_active_precise'] = precise_param_count_open_lm(big_df.width, big_df.depth)
    big_df['params_no_embed'] = precise_param_count_open_lm(big_df.width, big_df.depth, vocab_size=0)
    big_df['params_all'] = 12 * (big_df.width**2) * big_df.depth + (big_df.seq_len + 2 * big_df.vocab_size) * big_df.width
    
    # Counting FLOPs
    big_df['flops_per_token_att_no_embed'] = 6 * big_df['params_no_embed'] + 6 * big_df.seq_len * big_df.width * big_df.depth
    big_df['flops_per_token_att'] = 6 * big_df['params_active_precise']  + 6 * big_df.seq_len * big_df.width * big_df.depth
    big_df['flops_per_token_cc'] = precise_flops_per_token_chinchilla(big_df['width'], big_df['depth'])
    big_df['flops_per_token_no_att'] = 6 * big_df['params_active_precise']
    big_df['flops_per_token_no_att_no_embed'] = 6 * big_df['params_no_embed']
    big_df['flops_per_token'] = big_df['flops_per_token_no_att']

    big_df['params'] = big_df['flops_per_token'] / 6 
    big_df['eff_params_att'] = big_df['flops_per_token_att'] / 6


    big_df['train/loss_smoothed'] = apply_smoothing_filter(big_df, proportional_sliding_window_filter, compensate_for_logging_delay=True, key='train/loss')
    for k in big_df: 
        if k.startswith('train/') and k.endswith('_loss'):
            big_df[k + '_smoothed'] = apply_smoothing_filter(big_df, proportional_sliding_window_filter, compensate_for_logging_delay=False, key=k)
    return big_df

def process_sweep_df(big_df, trunc=None):
    big_df = big_df.copy()
    slim_cols = ['params', 'lr', 'bs', 'beta2', 'val/loss']
    if trunc is not None:
        slim_cols.extend([f'loss_mul_{M:.2f}' for M in trunc])
        for M in trunc:
            part = M / 20
            big_df = big_df.dropna(subset='train/loss_smoothed')
            # big_df[f'loss_mul_{M:.2f}'] = big_df[f'train/loss_smoothed'].apply(lambda x: x.iloc[len(x) - int(ind_trunc * len(x))])
            big_df[f'loss_mul_{M:.2f}'] = big_df[f'train/loss_smoothed'].apply(lambda x: x.iloc[int(part * len(x)) - 1])
    reduced_df = big_df[slim_cols].reset_index(drop=True).copy()
    
    if trunc is None:
        reduced_df = reduced_df.rename(columns=lambda x: x.replace('val/', ''))
        reduced_df = reduced_df.dropna(subset='loss')
        reduced_df['loss'] = reduced_df['loss'].apply(lambda x: x.iloc[-1])
    # print(reduced_df.columns)
    # reduced_df['loss_std'] = reduced_df['loss_std'].apply(lambda x: x.iloc[-1] if (len(x) > 0 and not trunc) else np.nan)
    
    reduced_df['lr'] = reduced_df['lr'].astype(float)
    reduced_df['bs'] = reduced_df['bs'].astype(float)
    reduced_df['beta2'] = reduced_df['beta2'].astype(float)
    reduced_df = reduced_df.sort_values('params')
    if trunc is None:
        reduced_df['excess_loss'] = reduced_df.groupby('params', group_keys=False).loss.apply(lambda x: x - x.min())
    else:
        for M in trunc:
            reduced_df[f'excess_loss_{M:.2f}'] = reduced_df.groupby('params', group_keys=False)[f'loss_mul_{M:.2f}'].apply(lambda x: x - x.min())
        reduced_df.drop('val/loss', axis=1, inplace=True)

    return reduced_df
