from configs import *
import pandas as pd

def extract_CI(fit_dict, flop_vals, key='n', conf_level = 0.05): # as in figure 1
    flops_grid = np.geomspace(np.min(flop_vals), np.max(flop_vals), 20)

    fit_dicts_bootstrap = fit_dict.get('bootstrap_weighted', None)
    key_coef, key_exponent = f'{key}_coef', f'{key}_exponent'
    if fit_dicts_bootstrap is not None:
        fit_vals_bootstrap = [fd[key_coef] * flops_grid ** fd[key_exponent] for fd in fit_dicts_bootstrap]
        exponents_bootstrap = [fd[key_exponent] for fd in fit_dicts_bootstrap]

        conf_int_lower = np.quantile(fit_vals_bootstrap, conf_level / 2, axis=0)
        conf_int_upper = np.quantile(fit_vals_bootstrap, 1 - conf_level / 2, axis=0)
        exps_lower = np.quantile(exponents_bootstrap, conf_level / 2, axis=0)
        exps_upper = np.quantile(exponents_bootstrap, 1 - conf_level / 2, axis=0)
    if key == 'multiplier':
        return f"({int(min(conf_int_lower))}, {int(max(conf_int_upper))})"
    return f"({exps_lower:.2f}, {exps_upper:.2f})"


def extract_label(row):
    config = tuple()
    for field in ['dataset', 'hparams', 'warmup', 'decay', 'param_count', 'val']:
        config += (row[field],)
    exp_string = CONFIG_DICT_LABEL[config]
    if exp_string == '':
        return 'Kaplan Adjusted'
    return CONFIG_DICT_LABEL[config]


def add_zero(x):
    if len(x.replace('.','')) < 4:
        if '.' not in x:
            return x + '.0'
        return x + '0'* (4 - len(x.replace('.','')))
    if len(x.replace('.','')) == 5 and x[0] == '0':
        return x[:5]
    return x


def debug(x):
    print(x['bs_median_weighted']['n_r2'])

def results_table(summary_df, flop_vals, validation=True):
    table_df = summary_df.copy()
    table_df['label'] = table_df.apply(extract_label, axis=1)
    table_df['dataset'] = table_df.dataset.apply(lambda x: 'OpenWebText2' if 'owt2' in x else 'RefinedWeb')
    table_df['validation'] = table_df.val.str.contains('val') if validation else ~table_df.fit_args.str.contains('val')
    table_df['a_est'] = table_df.fit_results.apply(lambda x: f"{x['bs_median_weighted']['n_exponent']:.3g}")
    table_df['a_R2'] = table_df.fit_results.apply(lambda x: f"{x['bs_median_weighted']['n_r2']:.3g}")
    table_df['a_CI'] = table_df.fit_results.apply(lambda x: extract_CI(x, flop_vals=flop_vals))
    table_df['r_range'] = table_df.fit_results.apply(lambda x: extract_CI(x, flop_vals=flop_vals, key='multiplier'))

    sorted_df = table_df[['label', 'dataset', 'validation', 'a_est', 'a_R2', 'a_CI', 'r_range']].sort_values(['a_est'], ascending=False).reset_index(drop=True)
    if validation != 'all':
        sorted_df = sorted_df.query('validation').drop('validation', axis=1)
    else:
        sorted_df = sorted_df.drop('validation', axis=1)

    grouped = sorted_df.groupby('dataset')

    max_len = max(len(group) for _, group in grouped)

    interleaved_rows = []
    for i in range(max_len):
        for _, group in grouped:
            if i < len(group):
                interleaved_rows.append(group.iloc[i])

    interleaved_df = pd.DataFrame(interleaved_rows).reset_index(drop=True)
    interleaved_df['a_est'] = interleaved_df.apply(lambda x: f"{x['a_est']} {x['a_CI']}", axis=1)

    interleaved_df = interleaved_df.drop('a_CI', axis=1)
    interleaved_df =  interleaved_df.rename({'label': 'Experiment', 'dataset': 'Dataset', 'a_est': '$a$ estimation', 'a_R2': '$R^2$ of $a$ fit', 'r_range': '$\rho^\star$ range'}, axis=1)
    return interleaved_df


def tuned_hparams(tuned_setting):
    table_df = tuned_setting.copy()
    table_df['Batch size'] = table_df.bs
    table_df['Learning rate'] = table_df.lr.apply(lambda x: f'{x:.2g}')
    table_df['\beta_2'] = table_df.beta2
    table_df['$\params$'] = table_df.params.apply(lambda x: f'{int(x/1e6)}')
    table_df = table_df[['$\params$','Learning rate', 'Batch size', '\beta_2', ]].reset_index(drop=True).drop_duplicates()
    
    # Generate the LaTeX file
    return table_df.sort_values('$\params$', ascending=False) #.to_latex(index=False, buf='paper/tables/hparams_tmp.tex')


def archs_table(one_setting):
    flops_per_token_cols = [col for col in one_setting.columns if 'flops_per' in col]

    N_exact = """5175744
    7508224
    9816640
    15608768
    22506048
    28695680
    37092096
    57432000
    84852864
    108540160
    149148032
    221014144
    347269120
    455546560
    612234304
    902090176"""

    # Convert N_exact to a list of integers
    N_exact_list = list(map(int, N_exact.split()))

    # Create the archs_table
    archs_table = one_setting[['depth', 'width', 'params_active_precise', 'params_no_embed'] + flops_per_token_cols].drop_duplicates().reset_index(drop=True).sort_values('params_active_precise')

    # Add N_exact as a new column in archs_table
    archs_table['N'] = archs_table.params_active_precise
    archs_table['N_exact'] = N_exact_list
    archs_table['N_att'] = (archs_table.flops_per_token_att / 6).astype(int)
    archs_table['N_no_head'] = archs_table.params_no_embed
    # archs_table['N_Chinchilla'] = (archs_table.flops_per_token_cc / 6).astype(int)
    archs_table = archs_table[['depth', 'width', 'N', 'N_exact', 'N_att', 'N_no_head']]

    # Define format_diff function
    def format_diff(value, exact):
        change_in_pct = ((value - exact) / exact) * 100
        return add_zero(f'{value/1e6:.4g}') + f' ({change_in_pct:+.1f}%)'

    # Apply the format_diff function separately for each column
    for col in ['N_exact', 'N_att', 'N_no_head']:
        archs_table[col] = [format_diff(row[col], row['N']) for idx, row in archs_table.iterrows()]

    
    archs_table.N = archs_table.N.apply(lambda x: add_zero(f'{x/1e6:.4g}'))

    rename_dict = {
    'depth': 'Depth',
    'width': 'Width',
    }

    # Generate the LaTeX file
    archs_table.rename(columns=rename_dict) #.to_latex(index=False, buf='paper/tables/archs_tmp.tex')

    return archs_table
    # with open('paper/tables/archs_tmp.tex', 'r') as file:
    #     tex_content = file.read()

    # tex_content = tex_content.replace('N\_exact', r'$\params_\text{Exact}$')
    # tex_content = tex_content.replace('N\_att', r'Effective $\params$ w/ attention')
    # tex_content = tex_content.replace('N\_no\_head', r'$\params$ w/o head')
    # tex_content = tex_content.replace('N', r'$\params$')

    # with open('paper/tables/archs.tex', 'w') as file:
    #     file.write(tex_content)


def format_a_estimation(value):
    value = value.replace('$', '\\$').replace('\\', '\\\\')
    formatted_value = ' '.join(
        [f"\\num{{{v}}}" if v.replace('.', '', 1).isdigit() else v for v in value.split()]
    )
    formatted_value = formatted_value.replace('(', '\\text{\\scriptsize(').replace(')', ')}')
    return formatted_value

def dataframe_to_latex(df: pd.DataFrame, filepath: str):
    df['$a$ estimation'] = df['$a$ estimation'].apply(format_a_estimation)
    latex_str = df.to_latex(escape=False, index=False)
    with open(filepath, 'w') as f:
        f.write(latex_str)