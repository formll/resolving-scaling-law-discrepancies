from plotting import *
from utils import *
from configs import *
import pandas as pd

from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib

def figure1(summary_df: pd.DataFrame, configs_to_show=None, save=False, ylim=None, arrows=True, save_path='../paper/figures/figure1.pdf'):
    # Figure 1
    if configs_to_show is None:
        configs_to_show = FIGURE1_CONFIGS
    fig, axes = plt.subplots(2, 3, figsize=[12, 8], facecolor='w')
    axes = axes.flatten()
    cfg_ind_to_axes_ind = [0, 1, 2, 5, 4]
    cfg_ind_to_letter = ['a', 'b', 'c', 'd', 'e']
    with plt.rc_context({'text.usetex': True,
                             'font.family': 'sans-serif',
                             'font.size': 14,  # For the text
                             'axes.titlesize': 18,  # For the subplot titles
                             'axes.labelsize': 20,  # For the x and y labels
                             'xtick.labelsize': 16,  # For the x tick labels
                             'ytick.labelsize': 16,  # For the y tick labels
                             'legend.fontsize': 12,  # For the legend
                             'figure.titlesize': 20,  # For the figure title
                             }):
        for i_cfg, i_axes in enumerate(cfg_ind_to_axes_ind):
            plt.sca(axes[i_axes])
            data, optimal_pairs, fit_results = summary_df.iloc[i_cfg][['data', 'optimal_pairs', 'fit_results']]
        
            legend_handles, legend_labels = opt_param_vs_compute_plot(data, optimal_pairs,
                                                                        fit_results['bs_median_weighted'], key='n',
                                                                        plot_error_bars=True, conf_level=0.05,
                                                                        fit_dicts_bootstrap=fit_results['bootstrap_weighted'], print_fit_as_text=True,
                                                                        return_legend=True, pct_in_latex_string=True,
                                                                        obs_color=CONFIG_DICT_COLOR[ configs_to_show[i_cfg]])
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, linewidth=0, pad=0.025)
            ax = axes[i_axes]
            # Place a text box in upper left in axes coords
            ax.text(0.9, 0.15, r'\textrm{\textbf{' + cfg_ind_to_letter[i_cfg] + r'}}', transform=ax.transAxes,  fontsize=26,
                    verticalalignment='top', bbox=props)

            plt.title(CONFIG_DICT_LABEL[configs_to_show[i_cfg]], color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]])
            if i_axes in [0, 4]:
                plt.ylabel(r'$N^{\star}(C)$', fontsize=14)
            if i_axes in [0, 4, 5]:
                plt.xlabel(r'Compute $C$ [FLOPs]', fontsize=14)
            if ylim is not None:
                plt.ylim(ylim)

        ax = axes[3]
        ax.axis('off')  # Turn off axis
        legend_labels[1] = 'Power law fit'


        ax.legend(handles=legend_handles, labels=legend_labels, loc='center', fontsize=18)
        plt.tight_layout()
        fig.canvas.draw()

        if arrows:
            arrows = [
            (0, 1, (0.8, 0.5), (0.2, 0.5)),  # From middle right of subplot 1 to middle left of subplot 2
            (1, 2, (0.8, 0.5), (0.2, 0.5)),  # From middle right of subplot 2 to middle left of subplot 3
            (2, 4, (0.5, 0.1), (0.7, 1.075)),  # From bottom middle of subplot 3 to top middle of subplot 5
            (2, 5, (0.5, 0.1), (0.5, 1.075))  # From bottom middle of subplot 3 to top middle of subplot 6
            ]
            plot_arrows(axes, arrows)

        if save:
            plt.rcParams.update({'text.usetex': True})
            plt.savefig(save_path, bbox_inches='tight')


def warm_evidence_figure(summary_df: pd.DataFrame, configs_to_show=None, save=False, save_path='../paper/figures/warmup_evidence.pdf'):
    if configs_to_show is None:
        configs_to_show = WARMUP_EVIDENCE_CONFIGS

    fig, axes = plt.subplots(1, 2, figsize=[10, 3], facecolor='w')
    axes = axes.flatten()

    titles = ['Kaplan et al. warmup', 'Our warmup']
    ylim = [1e7, 1e10]
    xlim = [1e16, 4e19]
    flop_vals = np.array(xlim)

    for i_cfg, ax in enumerate(axes):
        plt.sca(ax)
        data, optimal_pairs, fit_results = summary_df.iloc[i_cfg][
            ['data', 'optimal_pairs', 'fit_results']]
        with plt.rc_context({'text.usetex': True,
                             'font.family': 'serif',
                             'font.size': 16,  # For the text
                             'axes.titlesize': 18,  # For the subplot titles
                             'axes.labelsize': 16,  # For the x and y labels
                             'xtick.labelsize': 14,  # For the x tick labels
                             'ytick.labelsize': 14,  # For the y tick labels
                             'legend.fontsize': 16,  # For the legend
                             'figure.titlesize': 20,  # For the figure title
                             }):
            legend_stuff = opt_param_vs_compute_plot(data, optimal_pairs,
                                                     fit_results['bs_median_weighted'], key='t',
                                                     plot_error_bars=True,
                                                     fit_dicts_bootstrap=fit_results[
                                                         'bootstrap_weighted'], print_fit_as_text=False,
                                                     return_legend=True, pct_in_latex_string=True,
                                                     obs_color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]], flop_grid_endpoints=xlim)

            if i_cfg == 0:
                plt.fill_between(flop_vals, 3000 * 2 ** 19 * np.ones(2), color='tab:brown', alpha=0.25,
                                 label='Warmup period')
            else:
                warmup_vals = fit_results['bs_median_weighted']['n_coef'] * (flop_vals) ** \
                              fit_results['bs_median_weighted']['n_exponent']
                plt.fill_between(flop_vals, warmup_vals, color='tab:brown', alpha=0.25, label='Warmup period')
                handles, labels = plt.gca().get_legend_handles_labels()
                labels[1] = 'Power law fit'
                plt.legend(handles=handles, labels=labels, bbox_to_anchor=[1.01, 1.05], loc='upper left', fontsize=13)

            plt.title(titles[i_cfg], color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]])
            if i_cfg == 0:
                plt.ylabel(r'$D^{\star}(C)$', fontsize=14)
            plt.xlabel(r'Compute $C$ [FLOPs]', fontsize=14)

            plt.ylim(ylim)
            plt.xlim(xlim)

    plt.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches='tight')


def isoflop_loss_figure(summary_df, save=False, configs_to_show=None, save_path='../paper/figures/IsoFLOP-curves.pdf', ylim=[2.8, 7], min_multiplier=0.5):
    if configs_to_show is None:
        configs_to_show = FIGURE1_CONFIGS
    fig, axes = plt.subplots(2, 3, figsize=[16, 10], facecolor='w')
    axes = axes.flatten()
    cfg_ind_to_axes_ind = [0, 1, 2, 5, 4]
    max_max_loss = 0
    min_min_loss = 1e9
    with plt.rc_context({'text.usetex': False,
                         'font.family': 'sans-serif',
                         'font.size': 16,  # For the text
                         'axes.titlesize': 18,  # For the subplot titles
                         'axes.labelsize': 16,  # For the x and y labels
                         'xtick.labelsize': 14,  # For the x tick labels
                         'ytick.labelsize': 14,  # For the y tick labels
                         'legend.fontsize': 14,  # For the legend
                         'figure.titlesize': 20,  # For the figure title
                         }):
        for i_cfg, i_axes in enumerate(cfg_ind_to_axes_ind):
            plt.sca(axes[i_axes])
            data, optimal_pairs = summary_df.iloc[i_cfg][['data', 'optimal_pairs']]

            min_loss_plot, max_loss_plot = isoflop_curves_plot(data, optimal_pairs,  return_min_max_loss=True, min_multiplier=min_multiplier)
            ax = axes[i_axes]

            plt.title(CONFIG_DICT_LABEL[configs_to_show[i_cfg]], color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]])
            if i_axes in [0, 4]:
                plt.ylabel(r'Loss', fontsize=16)
            if i_axes in [0, 4, 5]:
                plt.xlabel(r'$N$', fontsize=16, family='sans-serif')
            max_max_loss = max(max_max_loss, max_loss_plot)
            min_min_loss = min(min_min_loss, min_loss_plot)
        for i_cfg, i_axes in enumerate(cfg_ind_to_axes_ind):
            plt.sca(axes[i_axes])
            plt.ylim(ylim)
        optimal_handle = [
            plt.Line2D([0], [0], color='k', marker='*', linestyle='', markersize=20, markeredgecolor='k', alpha=0.25),
            plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.7)  # Gray dashed line handle
        ]
        optimal_label = [r'Optimal $N$ in interpolated curve', 'Interpolated isoflop curve']
        ax = axes[3]
        ax.axis('off')  # Turn off axis
        ax.legend(handles=optimal_handle, labels=optimal_label, loc='lower center')
        cmap = plt.get_cmap('cool')
        norm = matplotlib.colors.Normalize(vmin=data.flops.min(), vmax=data.flops.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.36, pad=0.0)
        cbar.set_label(r'Compute $C$ [FLOPs]', fontsize=14)
        cbar.set_ticks([data.flops.min(), data.flops.max()])
        cbar.set_ticklabels([f'{data.flops.min():.2e}', f'{data.flops.max():.2e}'])
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()

        if save:
            plt.savefig(save_path, bbox_inches='tight')


def full_results_figure(summary_df,
                          fit_dict_weighted=None,
                          plot_error_bars=True,
                          fit_dicts_bootstrap=None,
                          plot_bootstrap_obvs=False,
                          conf_level=0.05,
                          configs_to_show=None,
                          save=None,
                          kaplan_adjusted=False,
                          save_path=f'../paper/figures/rw-results-fig.pdf'):
    summary_df = summary_df.copy()
    if configs_to_show is None:
        configs_to_show = FIGURE1_CONFIGS
    n_cols = 3
    n_rows = len(configs_to_show)
    xlim = [1e16, 2.6e19]
   
    with plt.rc_context({'font.family': 'sans-serif',
                            'text.usetex': True,
                            'font.size': 20, # For the text
                            'axes.titlesize': 28, # For the subplot titles
                            'axes.labelsize': 26, # For the x and y labels
                            'xtick.labelsize': 22, # For the x tick labels
                            'ytick.labelsize': 22, # For the y tick labels
                            'legend.fontsize': 28, # For the legend
                            'figure.titlesize': 28}
                            ):
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=[6*n_cols, 5*n_rows])
        max_max_rho = 0
        min_min_rho = 1e10
        min_min_N = 1e12
        max_max_N = 0
        min_min_loss = 1e14
        max_max_loss = 0
        for i_cfg, cfg in enumerate(configs_to_show):
            data, optimal_pairs, fit_results = summary_df.iloc[i_cfg][['data', 'optimal_pairs', 'fit_results']]
            fit_dict = fit_results['bs_median_weighted']
            fit_dicts_bootstrap = fit_results['bootstrap_weighted']
            # TODO: optimal pairs is essentially a subset of data - no need to have both!
            if n_rows > 1:
                plt.sca(axes[i_cfg,0])
            else:
                plt.sca(axes[0])
            
            opt_param_vs_compute_plot(data, optimal_pairs, fit_dict, 
                                      key='multiplier',
                                      fit_dicts_bootstrap=fit_dicts_bootstrap,
                                      plot_error_bars=True,
                                      return_legend=True,
                                      obs_color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]],
                                      big_font_conf_int=True,
                                      print_fit_as_text=True,
                                      conf_level=conf_level, pct_in_latex_string=True,
                                      kaplan_adjusted=kaplan_adjusted
            )
            if i_cfg == 0 and not kaplan_adjusted:
                plt.title(r"$\rho^\star$", color='k')
            max_max_rho = max(max_max_rho, plt.gca().get_ylim()[1])
            min_min_rho = min(min_min_rho, plt.gca().get_ylim()[0])
            if not kaplan_adjusted:
                plt.ylabel(CONFIG_DICT_LABEL[configs_to_show[i_cfg]], color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]])
            if i_cfg == n_rows-1:
                plt.xlabel('Compute $C$ [FLOPs]', fontfamily='sans-serif')

            if n_rows > 1:
                plt.sca(axes[i_cfg,1])
            else:
                plt.sca(axes[1])
            legend_handles, legend_labels = opt_param_vs_compute_plot(data, optimal_pairs, fit_dict,
                                                                      key='n', fit_dict_weighted=fit_dict_weighted,
                                                                      fit_dicts_bootstrap=fit_dicts_bootstrap,
                                                                      plot_error_bars=plot_error_bars,
                                                                      conf_level=conf_level,
                                                                      plot_bootstrap_obvs=plot_bootstrap_obvs,
                                                                      return_legend=True,
                                                                      print_fit_as_text=True,
                                                                      obs_color=CONFIG_DICT_COLOR[
                                                                          configs_to_show[i_cfg]],
                                                                      big_font_conf_int=True, pct_in_latex_string=True,
                                                                      kaplan_adjusted=kaplan_adjusted
            )
            if i_cfg==0 and not kaplan_adjusted:
                plt.title(r"$N^\star$", color='k', fontfamily="sans-serif")
            min_min_N = min(min_min_N, plt.gca().get_ylim()[0])
            max_max_N = max(max_max_N, plt.gca().get_ylim()[1])
            if i_cfg == n_rows-1:
                plt.xlabel('Compute $C$ [FLOPs]', fontfamily="sans-serif")

            if n_rows > 1:
                plt.sca(axes[i_cfg,2])
            else:
                plt.sca(axes[2])
            opt_param_vs_compute_plot(data, optimal_pairs, fit_dict,
                                      key='t', fit_dict_weighted=fit_dict_weighted,
                                      fit_dicts_bootstrap=fit_dicts_bootstrap,
                                      plot_error_bars=plot_error_bars,
                                      conf_level=conf_level,
                                      plot_bootstrap_obvs=plot_bootstrap_obvs,
                                      return_legend=True,
                                      print_fit_as_text=True,
                                      obs_color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]],
                                      big_font_conf_int=True, pct_in_latex_string=True,
                                      kaplan_adjusted=kaplan_adjusted
            )
            if i_cfg==0 and not kaplan_adjusted:
                plt.title(r"$D^\star$", color='k')
            if i_cfg == n_rows-1:
                plt.xlabel('Compute $C$ [FLOPs]', fontfamily="sans-serif")
            min_min_loss = min(min_min_loss, plt.gca().get_ylim()[0])
            max_max_loss = max(max_max_loss, plt.gca().get_ylim()[1])

            
            ax = axes[i_cfg,2] if n_rows > 1 else axes[2]
            flop_vals = np.array(xlim)
            if cfg[2] == 'long':
                plt.fill_between(flop_vals, y1=3000 * 2 ** 19 * np.ones(2), color='tab:brown', alpha=0.25, label='Warmup period')
            else:
                warmup_vals = fit_results['bs_median_weighted']['n_coef'] * (flop_vals) ** \
                              fit_results['bs_median_weighted']['n_exponent']
                plt.fill_between(flop_vals, warmup_vals, color='tab:brown', alpha=0.25, label='Warmup period')
            plt.xlim(flop_vals)

        for i_cfg, cfg in enumerate(configs_to_show):
            if n_rows > 1:
                plt.sca(axes[i_cfg,0])
                plt.ylim([min_min_rho, max_max_rho])
                plt.sca(axes[i_cfg,1])
                plt.ylim([min_min_N, max_max_N])
                plt.sca(axes[i_cfg,2])
                plt.ylim([min_min_loss, max_max_loss])
        legend_labels[1] = 'Power law fit'
        warmup_handle = matplotlib.patches.Patch(color='tab:brown', linestyle='-', alpha=0.25)
        warmup_label = 'Warmup period'
        legend_labels.append(warmup_label)
        legend_handles.append(warmup_handle)
        fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(legend_handles)//2)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.2)
        if save:
            if not kaplan_adjusted:
                plt.savefig(save_path.replace('rw', cfg[0]), bbox_inches='tight')
            else:
                plt.savefig(f'../paper/figures/kaplan-adjusted.pdf', bbox_inches='tight')


def opt_N_with_attention_figure(summary_df, configs_to_show=None, ylim=None, save=False, save_path='../paper/figures/accounting-att.pdf'):
    summary_df = summary_df.copy()
    if configs_to_show is None:
        configs_to_show = ATTENTION_ACCOUNTING_CONFIGS
    
    fig, axes = plt.subplots(2, 2, figsize=[8, 8], facecolor='w')
    axes = axes.flatten()
    
    with plt.rc_context({'text.usetex': True,
                            'font.family': 'serif',
                            'font.size': 16,  # For the text
                            'axes.titlesize': 18,  # For the subplot titles
                            'axes.labelsize': 16,  # For the x and y labels
                            'xtick.labelsize': 16,  # For the x tick labels
                            'ytick.labelsize': 16,  # For the y tick labels
                            'legend.fontsize': 16,  # For the legend
                            'figure.titlesize': 20,  # For the figure title
                            }):
        for i_cfg, cfg in enumerate(configs_to_show):
            plt.sca(axes[i_cfg])
            data, optimal_pairs, fit_results = summary_df.iloc[i_cfg][['data', 'optimal_pairs', 'fit_results']]
            legend_handles, legend_labels = opt_param_vs_compute_plot(data, optimal_pairs,
                                                                    fit_results['bs_median_weighted'], key='n',
                                                                    plot_error_bars=True, conf_level=0.05,
                                                                    fit_dicts_bootstrap=fit_results[
                                                                        'bootstrap_weighted'], print_fit_as_text=True,
                                                                    return_legend=True, pct_in_latex_string=True,
                                                                    obs_color=CONFIG_DICT_COLOR[cfg])

            plt.title(CONFIG_DICT_LABEL[cfg], color=CONFIG_DICT_COLOR[cfg])
            if i_cfg in [0, 2]:
                plt.ylabel(r'$N^{\star}(C)$')
            if i_cfg in [2,3]:
                plt.xlabel(r'Compute $C$ [FLOPs]')
            if ylim is not None:
                plt.ylim(ylim)
        legend_labels[1] = 'Power law fit'
        fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(legend_handles))
        plt.tight_layout()
    if save:
        plt.savefig(save_path, bbox_inches='tight')


def accuracy_vs_compute_figure(summary_df, plot_config=None, sweep_costs=None, xmax=4e20,
                               save=False, save_path='../paper/figures/accuracy_vs_compute.pdf'):
    if sweep_costs is None:
        sweep_costs = {'reduced sweep': 2.88e19, 'full sweep': 2.04e20}
    if plot_config is None:
        plot_config = dict(
                    # key='prediction_at_2.56e+19', #'exponent',
                    key='exponent', #'prediction_at_2.56e+19', #'exponent',
                    gt_value=0.5,
                    last_is_gt=False,
                    add_to_cost=0.0,
                    conf_level=0.05,
                    show_legend=False
                )
    with plt.rc_context({'font.size': 16,  # For the text
                         'axes.titlesize': 18,  # For the subplot titles
                         'axes.labelsize': 16,  # For the x and y labels
                         'xtick.labelsize': 12,  # For the x tick labels
                         'ytick.labelsize': 12,  # For the y tick labels
                         'legend.fontsize': 14,  # For the legend
                         'figure.titlesize': 20,  # For the figure title
                         }):
        fig, axs = plt.subplots(1, len(summary_df), facecolor='white', figsize=(5.5, 4))
        if len(summary_df) == 1:
            axs = [axs]

        for i in range(len(summary_df)):
            show_df = summary_df.iloc[i].results_df.copy()
            config = tuple(summary_df.iloc[i][['dataset', 'hparams', 'warmup', 'decay', 'param_count', 'val']])

            plt.sca(axs[i])
            legend_h, legend_l = compute_analysis_plot(show_df, **plot_config)
            if len(summary_df) > 1:
                plt.title(CONFIG_DICT_LABEL[config], color=CONFIG_DICT_COLOR[config])
            if xmax is not None:
                plt.xlim(show_df.index.min(), xmax)

        if plot_config.get('key', 'exponent'):
            axs[0].set_ylabel(f'$N^\star$ exponent $a$')
        else:
            val = plot_config['key'].split('_')[-1].replace('+','')
            axs[0].set_ylabel(f'$N^\star$({val})')

        styles = [':', '-.']
        for i, (name, cost) in enumerate(sweep_costs.items()):
            h = axs[-1].axvline(cost, ls=styles[i],
                                color=CONFIG_DICT_COLOR[config], lw=2, label=f'Cost of {name}')
            legend_h.append(h)
            legend_l.append(h.get_label())
            
        fig.legend(handles=legend_h, labels=legend_l, loc='upper left',
                   bbox_to_anchor=[0.975, 0.975])  

    plt.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches='tight')


def opt_loss_figure(summary_df, bootstrap_num=50,
                    save=False, save_path='../paper/figures/opt_loss.pdf'):
    with plt.rc_context({'font.size': 16,  # For the text
                         'axes.titlesize': 18,  # For the subplot titles
                         'axes.labelsize': 16,  # For the x and y labels
                         'xtick.labelsize': 14,  # For the x tick labels
                         'ytick.labelsize': 14,  # For the y tick labels
                         'legend.fontsize': 14,  # For the legend
                         'figure.titlesize': 20,  # For the figure title
                         'text.usetex': False}):
        fig, ax = plt.subplots(1, 1, figsize=[4, 4], facecolor='w')
        config_to_fit = tuple(summary_df.iloc[-1][['dataset', 'hparams', 'warmup', 'decay', 'param_count', 'val']])
        # iloc[-1] as we set the last config in summary_df to be the one to fit
        handles, labels = opt_loss_vs_compute_plot(summary_df, bootstrap_num=bootstrap_num,
                                                   fit_min_flop=1e16, fit_max_flop=5e17,
                                                   configs_to_fit=[config_to_fit], return_legend=True)

        plt.legend(handles=handles, labels=labels,
                   loc='upper left', bbox_to_anchor=[0.64, 1.01], framealpha=1)
    plt.ylim(2.9, 5.5)


    # plt.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches='tight')


def opt_loss_extended_figure(summary_df, bootstrap_num=50, 
                    save=False, save_path='../paper/figures/opt_loss_extended.pdf'):
    with plt.rc_context({'font.size': 16,  # For the text
                         'axes.titlesize': 18,  # For the subplot titles
                         'axes.labelsize': 16,  # For the x and y labels
                         'xtick.labelsize': 14,  # For the x tick labels
                         'ytick.labelsize': 14,  # For the y tick labels
                         'legend.fontsize': 14,  # For the legend
                         'figure.titlesize': 20,  # For the figure title
                         'text.usetex': False}):
        num_plots = len(summary_df)
        num_rows = 2
        num_cols = int(np.ceil(num_plots / num_rows))
        flop_vals = summary_df.iloc[0][['optimal_pairs']].iloc[0].flops
        xlim = [min(flop_vals), max(flop_vals)]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=[4 * num_cols, 4 * num_rows], facecolor='w')
        axes = axes.flatten()

        ax = axes[-1]

        cfg_ind_to_axes_ind = [0, 1, 2, 5, 4, 3]

        for k in range(num_plots):
            i = cfg_ind_to_axes_ind[k]
            i_row, i_col = i // num_cols, i % num_cols
            plt.sca(axes[i])
            config = tuple(summary_df.iloc[k][['dataset', 'hparams', 'warmup', 'decay', 'param_count', 'val']])
            # pdb.set_trace()
            handles, _ = opt_loss_vs_compute_plot(summary_df.iloc[[k]], bootstrap_num=bootstrap_num,
                                                    fit_min_flop=1e16, fit_max_flop=5e17,
                                                    configs_to_fit=[config], return_legend=True,
                                                    print_fit_as_text=True)
            plt.title(CONFIG_DICT_LABEL.get(config, '?!'), color=CONFIG_DICT_COLOR.get(config, 'k'))
            if i_col != 0:
                plt.ylabel('')
            if i_row > 0:
                axes[i-num_cols].set_xlabel('')
            plt.ylim(2.9, 5.5)
            plt.xlim(xlim)

    for k in range(num_plots, num_rows * num_cols):
        ax = axes[cfg_ind_to_axes_ind[k]]
        ax.axis('off')  # Turn off axis
    labels = ['Optimal loss', 'Saturating power law fit', 'Extrapolation']
    handles.append(matplotlib.patches.Patch(color='gray', alpha=0.2))
    labels.append('95% confidence region')

    ax.legend(handles=handles,
              labels=labels,
              loc='center', fontsize=14)
    for legend_handle in ax.get_legend().legendHandles:
        legend_handle.set_color('black')

    plt.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches='tight')


def hparams_fit_figure(reduced_df, reduced_df_opt_eta_and_bs, fit_dict,
                       min_params_for_fit=2.5e7, max_params_for_fit=1.1e8, excess_loss_thresh=0.03,
                       save=False, save_path='../paper/figures/hparams_fit.pdf'):
    show_keys = ['bs', 'lr']

    with plt.rc_context({'font.size': 16,  # For the text
                         'axes.titlesize': 18,  # For the subplot titles
                         'axes.labelsize': 16,  # For the x and y labels
                         'xtick.labelsize': 12,  # For the x tick labels
                         'ytick.labelsize': 12,  # For the y tick labels
                         'legend.fontsize': 14,  # For the legend
                         'figure.titlesize': 20,  # For the figure title
                         }):
        fig, axs = plt.subplots(1, 2, facecolor='white', figsize=(14, 5))
        y_ticks_dict = dict(lr=reduced_df['lr'].unique(), bs=reduced_df['bs'].astype(int).unique(),
                            beta2=[0.95, 0.99, 0.999])
        x_ticks = np.concatenate([reduced_df['params'].unique(), [901726208]])
        legends = []
        for k, show_key in enumerate(show_keys):
            show_df = reduced_df_opt_eta_and_bs.reset_index().copy()
            plt.sca(axs[k])
            legends.append(plot_sweep_key(show_df, reduced_df, show_key, fit_dict[show_key],
                                           x_ticks,
                                           y_ticks_dict, excess_loss_thresh,
                                           min_params_for_fit, max_params_for_fit, return_legend=True)
            )
        handles = legends[-1][0]
        labels = legends[0][1]
        num_betas = reduced_df.beta2.nunique()
        labels[num_betas] = 'Interpolated optimal values'
        for i in range(num_betas+2, num_betas+2+3):
            name, fit = labels[i].split(':')
            _, another_fit = legends[1][1][i].split(':')
            labels[i] = f'{name}:\n  {fit}\n  {another_fit}'
        handles.append([])
        legends.append([])
        leg = plt.legend(labels=labels, handles=handles, loc='upper left',
                         edgecolor='none', framealpha=0,
                         bbox_to_anchor=[1, 1.025],
                         bbox_transform=plt.gca().transAxes)
        for i, text in enumerate(leg.get_texts()):
            if i >= 5:
                text.set_verticalalignment('center')

    plt.tight_layout()
    if save:
        plt.savefig(save_path, bbox_inches='tight')


def full_sweep_figure(df, save=False):
    with plt.rc_context({'font.family': 'sans-serif',
        'font.size': 20, # For the text
            'axes.titlesize': 26, # For the subplot titles
            'axes.labelsize': 24, # For the x and y labels
            'xtick.labelsize': 20, # For the x tick labels
            'ytick.labelsize': 20, # For the y tick labels
            'legend.fontsize': 24, # For the legend
            'figure.titlesize': 24}): # For the figure title
        num_cols = 4
        num_rows = 2
        total_plots = 7  # 4 in the first row, 3 in the second row

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 10))
        axs = axs.flatten() 
        min_loss = df.iloc[:, 3:].min().min()

        beta2_values = df['beta2'].unique()
        colors = [get_color(i) for i in range(len(beta2_values))]
        color_map = dict(zip(beta2_values, colors))
        beta2_handles = [plt.Line2D([0], [0], color=color, marker='*', linestyle='', markersize=25) for color in colors]
        beta2_labels = [fr'$\beta_2$ = {beta2:.3g}' for beta2 in beta2_values]

        model_handles = []
        model_labels = []
        for idx, (name, group) in enumerate(df.groupby('bs')):
            if idx >= total_plots:  # Prevent more subplots than available
                break
            ax = axs[idx]
            model_sizes = [float(col.split('_')[-1]) for col in group.columns[3:]]
            sorted_indices = sorted(range(len(model_sizes)), key=lambda k: model_sizes[k])
            for i, col in enumerate(group.columns[3:]): 
                col_num = float(col.split('_')[-1])
                order_idx = sorted_indices.index(i)
                alpha_value = 0.5 + 0.5 * (order_idx / (len(model_sizes) - 1))
                model_handle, = ax.plot([], [], label=fr'$N$ = {col_num/1e6:.4g}M', marker=get_marker(i), color='gray', markersize=15, markerfacecolor='none')
                if fr'$N$ = {col_num/1e6:.4g}M' not in model_labels:
                    model_handles.append(model_handle)
                    model_labels.append(fr'$N$ = {col_num/1e6:.4g}M')
                for beta2, sub_group in group.groupby('beta2'):
                    if sub_group[col].isna().all():
                        continue
                    valid_idxs = ~sub_group[col].isna()
                    line, = ax.plot(sub_group.loc[valid_idxs, 'lr'], sub_group.loc[valid_idxs, col],
                                    label=f'{col_num/1e6:.2f}M Model, beta2={beta2}', marker=get_marker(i),
                                    color=color_map[beta2], markersize=15, alpha=1, markerfacecolor='none')
                    min_loss_idx = sub_group[col].idxmin()
                    ax.plot(sub_group.loc[min_loss_idx, 'lr'], sub_group.loc[min_loss_idx, col],
                            '*', color=line.get_color(), markersize=21)
            ax.set_title(f'Batch size = {int(name)}')
            if idx % num_cols == 0:
                ax.set_ylabel('Final Loss')
            
            if idx // num_cols > 0 or idx==num_cols - (total_plots%num_rows):
                ax.set_xlabel('Learning rate')
            ax.set_xscale('log')

            lrs_values = np.array(group['lr'].unique())
            lrs_ticks = [f"{lr:.1e}" for lr in lrs_values]
            ax.xaxis.set_major_locator(FixedLocator(lrs_values))

            ax.xaxis.set_major_formatter(FixedFormatter(lrs_ticks))
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')
                
            ax.grid(True)
            ax.set_ylim(min_loss - 0.05, 6.0)

        for i in range(total_plots, num_cols * num_rows):
            axs[i].axis('off')

        legend_ax = axs[total_plots]  # Use the first unused subplot for the legend
        legend_ax.axis('off')
        fig.legend(beta2_handles + model_handles, beta2_labels + model_labels, loc='center', bbox_to_anchor=(0.875, 0.25), fontsize='large')

        plt.tight_layout()
        if save:
            plt.savefig('../paper/figures/hparams-sweep.pdf')



def ideal_tuning_figure(summary_df, summary_compute, df_sweep_extended, tuning_data, optimal_pairs, fit_results, save=False, save_path=None, flop_vals_tuning=None):
    from matplotlib.gridspec import GridSpec
    if flop_vals_tuning is None:
        flop_vals_tuning = [optimal_pairs.c.min(), optimal_pairs.c.max()]
    with plt.rc_context({'text.usetex': False,
                         'font.family': 'sans-serif',
                         'font.size': 16,  # For the text
                         'axes.titlesize': 18,  # For the subplot titles
                         'axes.labelsize': 16,  # For the x and y labels
                         'xtick.labelsize': 14,  # For the x tick labels
                         'ytick.labelsize': 14,  # For the y tick labels
                         'legend.fontsize': 12,  # For the legend
                         'figure.titlesize': 20,  # For the figure title
                         }):
        fig = plt.figure(figsize=[10, 10], facecolor='w')
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        pos3 = ax3.get_position()
        ax3.set_position([pos3.x0, pos3.y0, pos3.width - 0.4, pos3.height])
        max_flop_ideal = optimal_pairs['c'].max()
        plt.sca(ax1)
        delta_losses_estimation_plot(df_sweep_extended)
        plt.sca(ax2)
        isoflop_curves_ideal_plot(summary_df, tuning_data)
         
        plt.sca(ax3)
        data_limited_compute = summary_compute.results_df.iloc[0].query("max_flop==@max_flop_ideal").bs_data.iloc[0]
        optimal_pairs_limited_compute = summary_compute.results_df.iloc[0].query("max_flop==@max_flop_ideal").optimal_pairs.iloc[0]
        fit_results_limited_compute = summary_compute.results_df.iloc[0].query("max_flop==@max_flop_ideal").fit_results.iloc[0]
        handles, lables = power_laws_with_tuning_data_plot(fit_results['bs_median_weighted'], fit_results['bs'], flop_vals_tuning=flop_vals_tuning, optimal_pairs_tuning=optimal_pairs,
                                optimal_pairs_limited_compute=optimal_pairs_limited_compute, data_limited_compute=data_limited_compute, fit_results_limited_compute=fit_results_limited_compute)
        plt.xlabel(r'Compute $C$ [FLOPs]')
        plt.ylabel(r'$N^{\star}(C)$')
        plt.legend(handles, lables, loc='lower right')
        plt.tight_layout()
        plt.show()
        if save:
            fig.savefig(save_path, bbox_inches='tight')


def seed_noise_figure(agg_df, save=False, save_path=None, xlim=(2.7, 7.5), ylim=(1e-3, 11e-2), remove_warmup=0.05):
    agg_df = agg_df.copy()
    seed_args = {'rw': RW_SEED_CONFIG, 'owt2': OWT2_SEED_CONFIG}
    titles = {'rw': 'RefinedWeb', 'owt2': 'OpenWebText2'}
    fig, axs = plt.subplots(1, 2, facecolor='white', figsize=(14, 5))
    with plt.rc_context({'text.usetex': False,
                            'font.family': 'sans-serif',
                            'font.size': 16,  # For the text
                            'axes.titlesize': 18,  # For the subplot titles
                            'axes.labelsize': 16,  # For the x and y labels
                            'xtick.labelsize': 16,  # For the x tick labels
                            'ytick.labelsize': 16,  # For the y tick labels
                            'legend.fontsize': 14,  # For the legend
                            'figure.titlesize': 20,  # For the figure title
                            }):
        for k, ds in enumerate(seed_args.keys()):
            plt.sca(axs[k])
            show_df = agg_df.query("dataset == @ds")
            plot_noise(show_df, xlim, remove_warmup=remove_warmup, title=titles[ds], **seed_args[ds])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.gca().invert_xaxis()
            plt.xlabel('Loss', fontsize=14)
            if k==0:
                plt.ylabel('Standard deviation of loss', fontsize=14)

        plt.plot([], [], ' ', label='Train loss', color='gray', linestyle='-')
        plt.plot([], [], 'o', label='Validation loss', color='gray', markeredgecolor='black')
        plt.plot([], [], ' ', label='Estimated validation sampling error', color='gray', linestyle='-.')
        plt.plot([], [], ' ', label='Bootstrap noise standard deviation', color='k', linestyle='--')
        plt.legend(loc='lower left', bbox_to_anchor=(1, 0.29))
        if save:
            plt.savefig(save_path, bbox_inches='tight')


def loss_curves_figure(df, save=False, configs_to_show=None, save_path='../paper/figures/loss-curves.pdf',remove_warmup=0, xlim=[1e15,2.7e19], ylim=[2.9, 6]):
    if configs_to_show is None:
        configs_to_show = FIGURE1_CONFIGS
    df = df.copy()
    fig, axes = plt.subplots(2, 3, figsize=[16, 10], facecolor='w')
    axes = axes.flatten()
    cfg_ind_to_axes_ind = [0, 1, 2, 5, 4]
    max_max_loss = 0
    min_min_loss = 1e9
    with plt.rc_context({'text.usetex': False,
                         'font.family': 'sans-serif',
                         'font.size': 16,  # For the text
                         'axes.titlesize': 18,  # For the subplot titles
                         'axes.labelsize': 16,  # For the x and y labels
                         'xtick.labelsize': 14,  # For the x tick labels
                         'ytick.labelsize': 14,  # For the y tick labels
                         'legend.fontsize': 14,  # For the legend
                         'figure.titlesize': 20,  # For the figure title
                         }):
        for i_cfg, i_axes in enumerate(cfg_ind_to_axes_ind):
            plt.sca(axes[i_axes])
            dataset, hparams, warmup, decay, param_count, _ = configs_to_show[i_cfg]
            df_setting = df.query("dataset == @dataset and hparams == @hparams and warmup == @warmup and decay == @decay").sort_values('params')
            plot_loss_curves(df_setting, remove_warmup, key='train/loss_smoothed', lw=0.5, alpha=1, param_count=param_count)
            plot_loss_curves(df_setting, remove_warmup, key='train/loss', lw=0.05, alpha=0.2, param_count=param_count)
            plt.xscale('log')
            plt.yscale('log')
            ax = axes[i_axes]

            plt.title(CONFIG_DICT_LABEL[configs_to_show[i_cfg]], color=CONFIG_DICT_COLOR[configs_to_show[i_cfg]])
            if i_axes in [0, 4]:
                plt.ylabel(r'Loss', fontsize=16)
            if i_axes in [0, 4, 5]:
                plt.xlabel(r'Compute $C$ [FLOPs]', fontsize=16, family='sans-serif')
            # max_max_loss = max(max_max_loss, max_loss_plot)
            # min_min_loss = min(min_min_loss, min_loss_plot)
        for i_cfg, i_axes in enumerate(cfg_ind_to_axes_ind):
            plt.sca(axes[i_axes])
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.grid(axis='x', which='major')
            plt.grid(axis='y', which='minor')

        ax = axes[3]
        ax.axis('off')  # Turn off axis

        params_list = df.sort_values('params').params.unique()
        colors = matplotlib.cm.cool(np.linspace(0, 1, len(params_list)))

        color_dict = {param: color for param, color in zip(params_list, colors)}
        legend_handles = [plt.Line2D([0], [0], color=color_dict[param], label=f'{int(param/1e6)}M') for param in params_list]
        ax.legend(handles=legend_handles, title=r'$N$', fontsize=14, title_fontsize=14, loc='lower center', ncol=2)
        plt.tight_layout()

        if save:
            plt.savefig(save_path, bbox_inches='tight')