import matplotlib.pyplot as plt
import numpy as np
from configs import *
from utils import *


def draw_obsvervations(optimal_pairs, key='n', plot_error_bars=False, obs_color='r'):
    key_std = f'{key}_star_std'
    if f'{key}_star_std' in optimal_pairs.columns and plot_error_bars:
        lower_bounds = optimal_pairs[key] - optimal_pairs[key] * np.exp(-optimal_pairs[key_std])
        upper_bounds = optimal_pairs[key] * np.exp(optimal_pairs[key_std]) - optimal_pairs[key]
        plt.errorbar(optimal_pairs.set_index('flops').index, optimal_pairs[key], yerr=[lower_bounds, upper_bounds],
                     fmt='o', color=obs_color, alpha=1, ms=6, label='Observations', capsize=6,
                     markeredgecolor='k', markeredgewidth=0.5)
    else:
        optimal_pairs.set_index('flops')[key].plot(style='x', color='r', alpha=1, ms=12, label='Observations')


def draw_fit(data, flops_grid, fit_dict, fit_dict_weighted, key='n', conf_level=0.05, plot_bootstrap_obvs=False, ci_color='grey', 
             obs_color='r', label_fit=True, fit_dicts_bootstrap=None, pct_in_latex_string=False):
    key_coef, key_exponent = f'{key}_coef', f'{key}_exponent'
    if fit_dicts_bootstrap is not None:
        # filter fit_dicts_bootstrap to only have dicts with keys containing key_coef
        fit_dicts_bootstrap = [fd for fd in fit_dicts_bootstrap if key_coef in fd]
        fit_vals_bootstrap = [fd[key_coef] * flops_grid ** fd[key_exponent] for fd in fit_dicts_bootstrap]
        exponents_bootstrap = [fd[key_exponent] for fd in fit_dicts_bootstrap]
        coefs_bootstrap = [fd[key_coef] * CHINCHILLA_FLOPS ** fd[key_exponent] for fd in fit_dicts_bootstrap]

        conf_int_lower = np.quantile(fit_vals_bootstrap, conf_level / 2, axis=0)
        conf_int_upper = np.quantile(fit_vals_bootstrap, 1 - conf_level / 2, axis=0)
        exponents_lower = np.quantile(exponents_bootstrap, conf_level / 2, axis=0)
        exponents_upper = np.quantile(exponents_bootstrap, 1 - conf_level / 2, axis=0)
        coefs_lower = np.quantile(coefs_bootstrap, conf_level / 2, axis=0)
        coefs_upper = np.quantile(coefs_bootstrap, 1 - conf_level / 2, axis=0)
        label_conf_region = f'{100*(1-conf_level):n}% confidence region' if not pct_in_latex_string else f'{100*(1-conf_level):n}\% confidence region'
        plt.fill_between(flops_grid, conf_int_lower, conf_int_upper, color=ci_color, alpha=0.2, label=label_conf_region)
        if plot_bootstrap_obvs:
            data.set_index('flops')[f'{key}_stars'].dropna().explode().groupby(level=0).sample(25).plot(style='xk', markersize=3,
                                                                                                        alpha=0.5)

    fit_colors = [obs_color, 'g']
    for i, (name, d) in enumerate(dict(Basic=fit_dict, Weighted=fit_dict_weighted).items()):
        if d is None:
            continue
        if label_fit:
            fit_label = f'{name} Fit: ${key} =$ {d[key_coef]:.3g} $C^{{{d[key_exponent]:.4g}}}$'
        else:
            fit_label = f'${key} =$ {d[key_coef]:.3g} $C^{{{d[key_exponent]:.4g}}}$'
        plt.plot(flops_grid, d[key_coef] * flops_grid ** (d[key_exponent]), '--', color=fit_colors[i], lw=2,
                    label=fit_label,)
    return exponents_lower, exponents_upper, coefs_lower, coefs_upper


def print_fit(key, big_font_conf_int, fit_dict, exponents_lower, exponents_upper, coefs_lower, coefs_upper):
    if exponents_upper is None:
        return
    key_coef, key_exponent = f'{key}_coef', f'{key}_exponent'
    if key == 'n':
        key_to_print = 'N^\star'
    elif key == 't':
        key_to_print = 'D^\star'
    else:
        key_to_print = r'\rho^\star'
    key_exponent_to_print = {'n':'$a$', 't':'$b$', 'multiplier':'$r$'}
    # Adding the fit parameters as text
    first_line = r'%s = $%.3g$ ' % (key_exponent_to_print[key], (fit_dict[key_exponent])) 
    if big_font_conf_int: # TODO: make this a parameter. having problems with formatting now
        conf = r'{\fontsize{18pt}{3em}\selectfont {$(%.2f, %.2f)$}}' % (exponents_lower, exponents_upper)
    else:
        conf = r'{\fontsize{12pt}{3em}\selectfont {$(%.2f, %.2f)$}}' % (exponents_lower, exponents_upper)
    second_line = r'$%s(%s)$ = %s ' % (key_to_print, CHINCHILLA_STR, fmt_model_size(fit_dict[key_coef] * CHINCHILLA_FLOPS ** fit_dict[key_exponent], key=key))
    if big_font_conf_int:
        conf2 = r'{\fontsize{17pt}{3em}\selectfont {(%s, %s)}}' % (fmt_model_size(coefs_lower, key=key), fmt_model_size(coefs_upper, key=key))
    else:
        conf2 = r'{\fontsize{11pt}{3em}\selectfont {(%s, %s)}}' % (fmt_model_size(coefs_lower, key=key), fmt_model_size(coefs_upper, key=key))
    exponent_ci_text = first_line + conf + '\n' + second_line + conf2
    props = dict(boxstyle='round', facecolor='white', alpha=0.7, linewidth=0)

    ax = plt.gca()
    ax.text(0.05, 0.95, exponent_ci_text, transform=ax.transAxes, #fontsize=10,
            verticalalignment='top', bbox=props)


def plot_existing_laws(key, flops_grid, kaplan_adjusted):
    if key == 'n':
        plt.plot(flops_grid, (flops_grid / (6 * 20)) ** 0.5, '-.', color='k', lw=2, label='Hoffmann law')
        if not kaplan_adjusted:
            plt.plot(flops_grid, 1.6e9 * (flops_grid / (1e15 * 24 * 60 * 60)) ** 0.88, ':', color='gray', lw=2, label='Kaplan law')
        else:
            plt.plot(flops_grid, 1.3e9 * (flops_grid / (8.64e19)) ** 0.73, ':', color='gray', lw=4, label="Adjusted Kaplan law")
    elif key == 't':
        plt.plot(flops_grid, (flops_grid / (6 / 20)) ** 0.5, '-.', color='k', lw=2, label='Hoffmann law')
        if not kaplan_adjusted:
            plt.plot(flops_grid, flops_grid / (1.6e9 * (flops_grid / (1e15 * 24 * 60 * 60)) ** 0.88) / 6, ':', color='gray', lw=2, label='Kaplan law')
        else:
            plt.plot(flops_grid, flops_grid / (1.3e9 * (flops_grid / (8.64e19)) ** 0.73) / 6, ':', color='gray', lw=4, label="adjusted Kaplan law")
        
    else:
        plt.plot(flops_grid, 20 * np.ones_like(flops_grid), '-.', color='k', lw=2, label='Hoffmann law')
        if not kaplan_adjusted:
            plt.plot(flops_grid, flops_grid / (1.6e9 * (flops_grid / (1e15 * 24 * 60 * 60)) ** 0.88)**2 / 6, ':', color='gray', lw=2, label='Kaplan law')
        else:
            plt.plot(flops_grid, flops_grid / (1.3e9 * (flops_grid / (8.64e19)) ** 0.73)**2 / 6, ':', color='gray', lw=4, label="Adjusted Kaplan law")


def draw_ideal_tuned_loss_curve(tuning_data_c, data_c, min_multiplier, i, c):
    if len(tuning_data_c) == 0:
        return
    tuning_data_c = tuning_data_c.iloc[0]
    # print(tuning_data_c['flops'])
    if tuning_data_c.loss_interp is None :
        return
    mask_fit_n = (c / (data_c['orig_n']**2) / 6 >= min_multiplier) & (c / (data_c['orig_n']**2) / 6 <= 30)
    loss_ = tuning_data_c['loss_orig']
    if len(loss_) == 0:
        return
    plt.scatter(data_c.orig_n[mask_fit_n], loss_[mask_fit_n], color='green', marker=get_marker(i), label=f'C={c:.4g}', s=15)
    loss_interp_ = tuning_data_c['loss_interp']
    mask_plot = (c / (tuning_data_c.n_interp**2) / 6 >= min_multiplier) & (c / (tuning_data_c.n_interp**2) / 6 <= 30)
    plt.plot(tuning_data_c.n_interp[mask_plot], loss_interp_[mask_plot], '--', color='green')
    if tuning_data_c['optimal_n'] is not None:
        plt.plot(tuning_data_c['optimal_n'][0], tuning_data_c['optimal_n'][1], '*', color='green', ms=12, markeredgecolor='k', alpha=0.5)


def opt_param_vs_compute_plot(data, optimal_pairs, fit_dict, key='n', fit_dict_weighted=None,
                              fit_dicts_bootstrap=None, plot_error_bars=False, conf_level=0.05,
                              plot_bootstrap_obvs=False, label_fit=True, return_legend=False,
                              print_fit_as_text=False, obs_color='r', flop_grid_endpoints=None, 
                              big_font_conf_int=False, kaplan_adjusted=False, ci_color='grey',pct_in_latex_string=False):
    flop_vals = data.flops.unique()
    enough_for_fit = len(optimal_pairs.query("~n.isna()")) >= 2

    if flop_grid_endpoints is None:
        flop_grid_endpoints = (np.min(flop_vals), np.max(flop_vals))
    flops_grid = np.geomspace(*flop_grid_endpoints, 20)
    draw_obsvervations(optimal_pairs, key=key, plot_error_bars=plot_error_bars, obs_color=obs_color)

    if enough_for_fit:
        exponents_lower, exponents_upper, coefs_lower, coefs_upper = draw_fit(data, flops_grid, fit_dict, fit_dict_weighted, fit_dicts_bootstrap=fit_dicts_bootstrap, key=key, conf_level=conf_level, plot_bootstrap_obvs=plot_bootstrap_obvs,
                    ci_color=ci_color, obs_color=obs_color, label_fit=label_fit, pct_in_latex_string=pct_in_latex_string)
    else:
        exponents_lower, exponents_upper, coefs_lower, coefs_upper = None, None, None, None

    plot_existing_laws(key, flops_grid, kaplan_adjusted)
        
    if print_fit_as_text and enough_for_fit and fit_dicts_bootstrap is not None:
        print_fit(key, big_font_conf_int, fit_dict, exponents_lower, exponents_upper, coefs_lower, coefs_upper)

    plt.ylim([optimal_pairs[key].min() * 0.7, optimal_pairs[key].max() * 1.3])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid('all')
    if not return_legend:
        plt.legend(loc='upper left', bbox_to_anchor=[0, 1])
    else:
        return plt.gca().get_legend_handles_labels()


def isoflop_curves_plot(data, optimal_pairs, return_min_max_loss=False, min_multiplier=None, color=None, tuning_data=None):
    colors_scale = plt.cm.cool(np.linspace(0.1, 1, len(data.flops.unique())))
    if color is not None:
        colors_scale = [color] * len(data.flops.unique())
    optimal_tuning_pairs = []
    for i, c in enumerate(data.flops.unique()):
        
        data_c = data.loc[data.flops == c].iloc[0]
        skip_optimal = False
        if len(optimal_pairs.loc[optimal_pairs.flops == c]) > 0:
            optimal_pairs_c = optimal_pairs.loc[optimal_pairs.flops == c].iloc[0]
        else:
            skip_optimal = True
        if data_c.n_interp is None:
            continue
        if (tuning_data is not None) and optimal_pairs_c['n'] > 220e6:
            continue
        # print(min_multiplier)
        if min_multiplier is not None:
            mask = c / (data_c['orig_n']**2) / 6 >= min_multiplier
            mask_interp = c / (data_c['n_interp']**2) / 6 >= min_multiplier
        else:
            mask = np.ones_like(data_c['orig_n'], dtype=bool)
            mask_interp = np.ones_like(data_c['n_interp'], dtype=bool)
        plt.scatter(data_c.orig_n[mask], data_c.orig_loss[mask], color=colors_scale[i], marker=get_marker(i), label=f'C={c:.4g}', s=15)
        # print(data_c['n_interp'])
        plt.plot(data_c.n_interp[mask_interp], data_c.loss_interp[mask_interp], '--', color=colors_scale[i])
        if tuning_data is not None:
            tuning_data_c = tuning_data.loc[tuning_data.flops == c]
            draw_ideal_tuned_loss_curve(tuning_data_c, data_c, min_multiplier, i, c)
            
        

        if not skip_optimal:
            plt.plot(optimal_pairs_c['n'], data_c['loss_interp'][int(data_c['opt_ind'])],
                     '*', ms=12, color=colors_scale[i], markeredgecolor='k', alpha=0.5)
    if len(optimal_tuning_pairs) > 1:
        fit = np.polyfit(np.log([x[0] for x in optimal_tuning_pairs]), np.log([x[1] for x in optimal_tuning_pairs]), 1)
        print(f'fit: {fit[0]:.2f}, {fit[1]:.2f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(axis='x', which='major')
    plt.grid(axis='y', which='minor')
    
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', which='minor', labelsize=12)

    if return_min_max_loss:
        return plt.gca().get_ylim()
    else:
        return plt.gca().get_legend_handles_labels()


def opt_loss_vs_compute_plot(summary_df, configs_to_fit=tuple(), 
                             fit_min_flop=1e16, fit_max_flop=5e17, conf_level=0.05,
                             return_legend=True, print_fit_as_text=False, bootstrap_num=None):
    for i, (_, row) in enumerate(summary_df.iterrows()):
        config = tuple(row[field] for field in ['dataset', 'hparams', 'warmup', 'decay', 'param_count', 'val'])
        optimal_pairs = row['optimal_pairs'].set_index('flops').dropna()
        if bootstrap_num is None:
            bootstrap_num = row['data'][['loss_stars']].dropna().applymap(len).min().min()
        
        
        
        flops_all = optimal_pairs.index.values
        # flops_all += [flops_all[-1]*1.2]
        flops = optimal_pairs.dropna().truncate(before=fit_min_flop, after=fit_max_flop).index.values

        row['optimal_pairs'].set_index('flops').loss.dropna().plot(
            logx=True, logy=True, style='-'+CONFIG_DICT_MARKER[config],
            label=CONFIG_DICT_LABEL[config],
            lw=0.5, color=CONFIG_DICT_COLOR[config], markersize=7, markerfacecolor='none'
        )

        extrap_flops = np.geomspace(0.005 * optimal_pairs.index.min(), 200 * optimal_pairs.index.max(), 100)

        if config in configs_to_fit:
            # A, E, alpha = summary_df.iloc[0].fit_results['bs_median'] in keys of ['A', 'E', 'alpha']
            A, E, alpha = summary_df.iloc[-1].fit_results['bs_median']['A'], summary_df.iloc[-1].fit_results['bs_median']['E'], summary_df.iloc[-1].fit_results['bs_median']['alpha']
            plt.plot(flops, E + A*(flops**-alpha), lw=1, color=CONFIG_DICT_COLOR[config], linestyle='-',
                     label=f'Fit, $L={A:.2f}C^{{-{alpha:.3f}}} + {E:.2f}$', marker=CONFIG_DICT_MARKER[config])
            plt.plot(extrap_flops, E + A*(extrap_flops**-alpha), label=f'Extrapolation', lw=1,
                     color=CONFIG_DICT_COLOR[config], linestyle='--')

            fit_dicts_bootstrap = [fd for fd in summary_df.iloc[-1].fit_results['bootstrap'] if 'alpha' in fd]
            fit_vals_bootstrap = [fd['E'] + fd['A'] * (extrap_flops ** -fd['alpha']) for fd in fit_dicts_bootstrap]
            conf_int_lower = np.quantile(fit_vals_bootstrap, conf_level / 2, axis=0)
            conf_int_upper = np.quantile(fit_vals_bootstrap, 1 - conf_level / 2, axis=0)
            plt.fill_between(extrap_flops, conf_int_lower, conf_int_upper, color=CONFIG_DICT_COLOR[config], alpha=0.2)

    if print_fit_as_text and configs_to_fit:
        A_fmt = f'{A:.2e}'.replace('+0', '').replace('e', '\mathrm{e}')
        fit_text = f'Fit: $L={A_fmt}\\cdot C^{{-{alpha:.3f}}} + {E:.2f}$'
        props = dict(boxstyle='round', facecolor='white', alpha=0.7, linewidth=0)
        ax = plt.gca()
        ax.text(0.95, 0.95, fit_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    x_lo = round_down_to_first_decimal(summary_df.optimal_pairs.apply(lambda x: x.flops.min()).min() * 0.8)
    x_hi = round_up_to_first_decimal(summary_df.optimal_pairs.apply(lambda x: x.flops.max()).max() * 1.2)
    plt.xlim(x_lo, x_hi)
    plt.ylabel('Estimated optimal loss $L^\star(C)$')
    plt.xlabel('Compute $C$ [FLOPs]')
    plt.grid(axis='y', which='minor')
    plt.grid(axis='x', which='major')
    if not return_legend:
        plt.legend(loc='upper left', bbox_to_anchor=[1, 1.01])
    else:
        return plt.gca().get_legend_handles_labels()


def compute_analysis_plot(results_df, key='exponent',
                          gt_value=None, last_is_gt=False, add_to_cost=0, conf_level=0.05, show_legend=False):
    show_df = results_df.copy()
    show_df.index = show_df.index + add_to_cost
    if last_is_gt:
        if key.startswith('prediction'):
            gt_value = show_df.iloc[-1].optimal_pairs.iloc[-1]['n']
        else:
            gt_value = show_df.iloc[-1].exponent

    plt.fill_between(show_df.index, show_df[key + '_lo'], show_df[key + '_hi'], color='gray', alpha=0.2,
                     label=f'${100 * (1 - conf_level):g}\%$ confidence region')
    plt.plot(show_df[key], '-k', lw=2, label='Point estimate')
    if key.startswith('prediction'):
        plt.yscale('log')
        plt.ylim([round_down_to_first_decimal(show_df[key + '_lo'].min()),
                  round_up_to_first_decimal(show_df[key + '_hi'].max())])


    if gt_value is not None:
        plt.axhline(gt_value, ls='--', color='k', lw=1, label='Nominal value')

    ax1 = plt.gca()
    handles, labels = ax1.get_legend_handles_labels()

    plt.xlabel('Compute $C$ [FLOPs]')

    if gt_value is not None:
        bootstrap_error = show_df.bs_predictions.apply(
            lambda x: (((x[key] - gt_value) / gt_value) ** 2).mean() ** 0.5)
        ax2 = ax1.twinx()
        ax2.plot(bootstrap_error, '-b', lw=1, label='RMS relative\n bootstrap error')
        plt.yscale('log')
        plt.ylim([10 ** np.floor(np.log10(bootstrap_error.min())),
                  10 ** np.ceil(np.log10(bootstrap_error.max()))])
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        # ax2.set_ylabel('Y axis label 2', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue', which='both')
        ax2.spines['right'].set_color('blue')

    plt.xscale('log')
    plt.xlim([show_df.index.min(), show_df.index.max()])


    if show_legend:
        plt.legend(labels=labels, handles=handles)
    else:
        return handles, labels


def plot_arrows(axes, arrows):
    # Define arrow properties
    arrowprops = dict(facecolor='black', edgecolor='black', width=6, headwidth=20, shrink=0.1)

    # Add arrows between the specified subplots
    for start_idx, end_idx, start_pos, end_pos in arrows:
        ax_start = axes.flatten()[start_idx]
        ax_end = axes.flatten()[end_idx]

        # Get positions in figure coordinates
        start_bbox = ax_start.get_position()
        end_bbox = ax_end.get_position()

        # Calculate arrow start and end points
        start_coord = (
        start_bbox.x0 + start_pos[0] * start_bbox.width, start_bbox.y0 + start_pos[1] * start_bbox.height)
        end_coord = (end_bbox.x0 + end_pos[0] * end_bbox.width, end_bbox.y0 + end_pos[1] * end_bbox.height)

        # Draw the arrow on the main figure
        ax_end.annotate(
            '',
            xy=end_coord, xytext=start_coord,
            arrowprops=arrowprops,
            xycoords='figure fraction', textcoords='figure fraction',
        )


def plot_sweep_key(show_df, reduced_df, show_key, fit_dict, x_ticks, y_ticks_dict, excess_loss_thresh, min_params_for_fit, max_params_for_fit, return_legend=False):
    handles = []
    labels = []
    for j, beta2_ in enumerate([0.95, 0.99, 0.999]):
        query = f'beta2 == {beta2_}' # & excess_loss < @excess_loss_thresh'
        sample_ = reduced_df.reset_index().query(query).sort_values('params')
        if len(sample_) > 0:
            plt.scatter(sample_['params'], sample_[show_key], alpha=np.maximum(0.01, 1 - sample_['excess_loss'].values / excess_loss_thresh),
                        marker=get_marker(j), label='data, $\beta_2$ = ' + str(beta2_), c=get_color(j), 
                        s=8 * (12 - 3 * j), edgecolors='k')
            handle = plt.Line2D([0], [0], marker=get_marker(j), color='w', markerfacecolor=get_color(j),
                                markersize=8, markeredgewidth=1, markeredgecolor='k')
            handles.append(handle)
            labels.append(fr'Grid points, $\beta_2 = {beta2_}$')      

    show_df.set_index('params')[show_key].plot(
        logx=True, logy=True, marker='d', markersize=10, label='Interpolated optimal {}'.format(show_key),
        lw=1.25, markeredgewidth=2, markeredgecolor=get_color(j+2), color=get_color(j+2), markerfacecolor='none')
    handles.append(plt.Line2D(
        [0], [0], marker='d', color='w', markerfacecolor='none', markersize=8,
        markeredgewidth=2, markeredgecolor=get_color(j+2)))
    labels.append('Interpolated optimal {}'.format(show_key.upper()))

    show_df_for_fit = show_df.query('params > @min_params_for_fit and params < @max_params_for_fit')
    plt.scatter(show_df_for_fit.params, show_df_for_fit[show_key], marker='d', s=200, c=get_color(j+2), label='Points used for fit', edgecolors=get_color(j+2), linewidths=0.2)
    handles.append(plt.Line2D([0], [0], marker='d', markersize=10, c=get_color(j+2), markeredgewidth=1, markeredgecolor=get_color(j+2)))
    labels.append('Points used for fit')

    
    x_vals = np.array([0.8 * x_ticks[0], 1.2 * x_ticks[-1]])
    fit_vals = fit_dict[show_key + '_coef'] * (x_vals ** fit_dict[show_key + '_exponent'])
    label_computed_fit = f'Fit: {show_key.upper()} = ${fit_dict[show_key + "_coef"]:.2g} N^{{{fit_dict[show_key + "_exponent"]:.3g}}} \ (R^2={fit_dict[show_key + "_r2"]:.3g})$'
    plt.plot(x_vals, fit_vals, '--k', 
                label=label_computed_fit)

    C_to_N_coef, C_to_N_exponent = 1 / (120 ** 0.5), 0.5  # Hoffmann

    N_to_C_coef = (1 / C_to_N_coef) ** (1 / C_to_N_exponent)
    N_to_C_exponent = 1 / C_to_N_exponent

    if show_key == 'bs':
        fit_vals_manual = 160.0 * ((x_vals/108e6) ** (2/3))
        label_manual_fit = r'Rounded fit: BS =  $160 (N / 108\mathrm{e}6)^{2/3}$'
        ds_C_coef, ds_C_exponent = 0.2920, 0.3271
        ds_N_coef, ds_N_exponent = ds_C_coef * (N_to_C_coef ** ds_C_exponent), N_to_C_exponent * ds_C_exponent

        ds_N_coef /= 2048  # to get BS in sequences instead of in tokens

        fit_deepseek = ds_N_coef * (x_vals ** ds_N_exponent)  # need to devide by sequence length
        label_deepseek_fit = f'DeepSeek fit: BS$ = {ds_N_coef:.2g} N^{{{ds_N_exponent:.3g}}}$'

    elif show_key == 'lr':
        fit_vals_manual = 0.0047 * ((x_vals/108e6) ** (-1/3))
        label_manual_fit = r'Rounded fit: LR =  $0.0047 (N / 108\mathrm{e}6)^{-1/3}$'

        ds_C_coef, ds_C_exponent = 0.3118, -0.1250

        ds_N_coef, ds_N_exponent = ds_C_coef * (N_to_C_coef ** ds_C_exponent), N_to_C_exponent * ds_C_exponent

        fit_deepseek = ds_N_coef * (x_vals ** ds_N_exponent)
        label_deepseek_fit = f'DeepSeek fit: LR $= {ds_N_coef:.2g} N^{{{ds_N_exponent:.3g}}}$'
    else:
        fit_vals_manual = None
    if fit_vals_manual is not None:
        plt.plot(x_vals, fit_vals_manual, ':r',
                    label=label_manual_fit)

        plt.plot(x_vals, fit_deepseek, linestyle='-.', color='orange',
                    label=label_deepseek_fit)

    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(x_ticks, labels=[f'{x//1e6:n}M' for x in x_ticks])
    plt.yticks(y_ticks_dict[show_key], labels=y_ticks_dict[show_key])
    plt.gca().set_yticks([], minor=True)
    plt.gca().set_xticks([], minor=True)
    plt.gca().set_axisbelow(True)
    plt.grid('major', color=[0.8, 0.8, 0.8, 1])
    # print(y_ticks_dict[show_key])
    # print(0.8 * min(y_ticks_dict[show_key]), max(1.2 * y_ticks_dict[show_key][-1], fit_vals.max()))
    plt.xlim(0.8 * x_ticks[0], 1.2 * x_ticks[-1])
    plt.ylim(0.8 * min(y_ticks_dict[show_key]), max(max(1.2 * y_ticks_dict[show_key]), fit_vals.max()))
    plt.xlabel(f'$N$')
    plt.title(f'{KEYS_TO_TITLE_SWEEP[show_key]}')

    handles = handles + [plt.Line2D([], [], linestyle='--', color='k'),
                                    plt.Line2D([], [], linestyle=':', color='r'),
                                    plt.Line2D([], [], linestyle='-.', color='orange')]
    labels = labels + [label_computed_fit, label_manual_fit, label_deepseek_fit]
    if return_legend:
        return handles, labels
    else:
        plt.legend(handles=handles, labels=labels)





def plot_noise(agg_df, xlim, remove_warmup=0.05, noise_low=0.002, noise_high=0.05, l_threshold_high=7, l_threshold_low=3, title=''):
    colors = matplotlib.cm.cool(np.linspace(0, 1, len(agg_df)))
    for i, (idx, row) in enumerate(agg_df.iterrows()):
        num = int(row['train/loss_smoothed_mean'].index.max() * remove_warmup)
        plt.plot(row['train/loss_smoothed_mean'].loc[num:], 
                row['train/loss_smoothed_std'].loc[num:], 
                color=colors[i], label=f"$N$ = {int(row['params']/1e6)}M")
    for i, (idx, row) in enumerate(agg_df.iterrows()):
        num = int(row['train/loss_smoothed_mean'].index.max() * remove_warmup)
        plt.plot(row['val/loss_mean'].loc[num:], row['val/loss_std'].loc[num:],
                    color=colors[i], lw=0, markeredgecolor='k', marker='o', ms=6)
        plt.axhline(row['val/loss_std_mean'].mean(), linestyle='-.', color=colors[i])

    x_values = np.linspace(*xlim, 500)

    def our_noise(x):
        if x < np.log(l_threshold_low):
            return np.log(noise_low)
        elif x > np.log(l_threshold_high):
            return np.log(noise_high)
        else:
            return np.interp(x, [np.log(l_threshold_low), np.log(l_threshold_high)], [np.log(noise_low), np.log(noise_high)])

    f_values = np.array([np.exp(our_noise(np.log(x))) for x in x_values])

    plt.plot(x_values, f_values, color='black', linestyle='--')
    plt.title(title)


def plot_loss_curves(df_setting, remove_warmup, key='train/loss_smoothed', lw=0.75, alpha=1, param_count='standard'):
    params_list = df_setting.params.unique()
    colors = matplotlib.cm.cool(np.linspace(0, 1, len(params_list)))
    param_to_col_dict = {params_list[i]:colors[i] for i in range(len(params_list))}
    df_setting = df_setting.copy()
    for idx, row in df_setting.iterrows():
        num = int(row[key].index.max() * remove_warmup)
        if param_count == 'standard':
            params = row['params']
        elif param_count == 'kaplan':
            params = row['params_no_embed']
        elif param_count == 'attention':
            params = row['eff_params_att']
        plt.plot(row[key].loc[num:].index*row['seq_len']*row['bs']*params*6 , row[key].loc[num:],
        color=param_to_col_dict[row['params']], label=f"$N$ = {int(row['params']/1e6)}M", lw=lw, alpha=alpha)


def opt_n_vs_compute_ideal_tuning_plot(tuning_fit_results, fit_results_bootstrap, optimal_pairs_tuning, flop_vals_tuning=[1.25e16, 8e17], conf_level=0.05):
    plt.plot(flop_vals_tuning, tuning_fit_results['n_coef']*(np.array(flop_vals_tuning))**tuning_fit_results['n_exponent'],
            color=get_color(8), linestyle='--')
    point_estimates = optimal_pairs_tuning.dropna()
    lower_bounds_tuning = point_estimates['n'] - point_estimates['n']*np.exp(-point_estimates['n_star_std'])
    upper_bounds_tuning = point_estimates['n']*np.exp(point_estimates['n_star_std']) - point_estimates['n']
    flops_grid = np.geomspace(flop_vals_tuning[0], flop_vals_tuning[1], 5000)
    fit_vals_bootstrap = [fd['n_coef'] * flops_grid ** fd['n_exponent'] for fd in fit_results_bootstrap]
    exponents_bootstrap = [fd['n_exponent'] for fd in fit_results_bootstrap]

    conf_int_lower = np.quantile(fit_vals_bootstrap, conf_level / 2, axis=0)
    conf_int_upper = np.quantile(fit_vals_bootstrap, 1 - conf_level / 2, axis=0)
    exponents_lower = np.quantile(exponents_bootstrap, conf_level / 2, axis=0)
    exponents_upper = np.quantile(exponents_bootstrap, 1 - conf_level / 2, axis=0)
    plt.fill_between(flops_grid, conf_int_lower, conf_int_upper, color=get_color(8), alpha=0.2, label=f'{100*(1-conf_level):n}% confidence region')
    plt.errorbar(point_estimates['c'], point_estimates['n'], yerr=[lower_bounds_tuning, upper_bounds_tuning],
                    fmt='s', color=get_color(8), alpha=1, ms=6, label='Estimated ideal tuning', capsize=6,
                    markeredgecolor='k', markeredgewidth=0.5)
    return exponents_lower, exponents_upper


def power_laws_with_tuning_data_plot(tuning_fit_results, fit_results_bootstrap, flop_vals_tuning=[1.25e16, 8e17], optimal_pairs_tuning=None,
                                      data_limited_compute=None, optimal_pairs_limited_compute=None, fit_results_limited_compute=None, conf_level=0.05):
    opt_param_vs_compute_plot(data_limited_compute, optimal_pairs_limited_compute, fit_results_limited_compute['bs_median_weighted'], key='n', plot_error_bars=True,
                                    fit_dicts_bootstrap=fit_results_limited_compute['bootstrap_weighted'], ci_color='red')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    # replace handles[0] with the same handle but with color gray
    handles = [handle for handle, _ in zip(handles, labels)]
    handles.pop(1) # remove the handle for the previous power law
    labels.pop(1) # remove the label for the previous power law
    grey_handle = handles[0]
    grey_handle = matplotlib.patches.Patch(color='grey', alpha=0.2)
    handles = [grey_handle] + handles[1:]

    exponents_lower, exponents_upper = opt_n_vs_compute_ideal_tuning_plot(tuning_fit_results, fit_results_bootstrap, optimal_pairs_tuning, 
    flop_vals_tuning=flop_vals_tuning, conf_level=conf_level)
    exponents_bootstrap_orig = [fd['n_exponent'] for fd in fit_results_limited_compute['bootstrap_weighted']]
    orig_exponents_lower = np.quantile(exponents_bootstrap_orig, conf_level / 2, axis=0)
    orig_exponents_upper = np.quantile(exponents_bootstrap_orig, 1 - conf_level / 2, axis=0)

    handles_2, labels_2 = plt.gca().get_legend_handles_labels()
    handles_2.insert(-1, matplotlib.lines.Line2D([0], [0], color='red', lw=2, linestyle='--'))
    labels_2.insert(-1, f"Power law fit:\n$a$ = {fit_results_limited_compute['bs_median_weighted']['n_exponent']:.3f} ({orig_exponents_lower:.2f}, {orig_exponents_upper:.2f})")
    handles_2.append(matplotlib.lines.Line2D([0], [0], color=get_color(8), lw=2, linestyle='--'))
    labels_2.append(f'Power law fit:\n$a$ = {tuning_fit_results["n_exponent"]:.3f} ({exponents_lower:.2f}, {exponents_upper:.2f})')
    plt.legend([],[], loc='upper left')
    # plt.xlim(flop_vals_tuning)
    return handles + [handle for handle, label in zip(handles_2, labels_2) if label not in labels and 'C^' not in label],\
     labels + [label for label in labels_2 if label not in labels and 'C^' not in label]


def isoflop_curves_ideal_plot(summary_df, tuning_data):
    data, optimal_pairs = summary_df.iloc[-1][['data', 'optimal_pairs']]
    isoflop_curves_plot(data, optimal_pairs,  return_min_max_loss=True, min_multiplier=2, tuning_data=tuning_data)
    handles = []
    labels = []
    handles.append(matplotlib.lines.Line2D([0], [0], color='green', lw=2, linestyle='--'))
    labels.append('Estimated loss with ideal tuning')

    plt.xlabel('Model size $N$')
    plt.ylabel('$L$')

    plt.legend(handles, labels)
    plt.grid('minor')


def delta_losses_estimation_plot(df_sweep_extended):
    unique_params = df_sweep_extended['params'].unique()

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(unique_params))
    cmap = plt.get_cmap('cool')
    for i, param in enumerate(unique_params):
        subset = df_sweep_extended[df_sweep_extended['params'] == param]
        plt.plot(subset['M'], subset['loss_diff_smoothed'], label=f"$N$ = {int(param/1e6)}M", color=cmap(norm(i)))
        plt.plot(subset['M'], subset['loss_diff'], color=cmap(norm(i)), alpha=0.2, linestyle='--')

    plt.xlabel(r'Token-to-parameter ratio $\rho$')
    plt.ylabel(r'Estimated $L - L^\star$', fontsize=18) 
    
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.grid('major', color=[0.8, 0.8, 0.8, 1])