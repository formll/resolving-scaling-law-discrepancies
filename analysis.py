import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from itertools import product
from configs import *
from utils import *


# Define the Huber loss function
def custom_huber_loss(y_true, y_pred, delta=1e-3):
    diff = y_true - y_pred
    cond = np.abs(diff) <= delta
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return np.sum(loss)

def huber_loss_objective(params, F, losses):
    a, e, alpha = params
    predictions = np.logaddexp(a - alpha * np.log(F), e)
    return custom_huber_loss(np.log(losses), predictions, delta=1e-3)


def fetch_flop(df, flop, loss_key='train/loss_smoothed', warmup_remove_factor=1e-12, n_key='params', 
               seq_len=2048, bs_key='bs', keep_bs_lr_keys=False,
               flop_per_token_key='flops_per_token', flop_tolerance=0.1):
    out = []
    for _, row in df.iterrows():
        if len(row[loss_key]) == 0:
            continue
        loss_vals = row[loss_key].dropna().groupby(level=0).mean().sort_index()
        step_vals = loss_vals.index
        mask = step_vals >= ((warmup_remove_factor * row.warmup_tokens) / row.bs / row.seq_len)
        loss_vals = loss_vals[mask]
        loss_vals.index = loss_vals.index.astype(float) * seq_len * row[bs_key] * row[flop_per_token_key]
        flop_vals = loss_vals.index
        
        if len(loss_vals) == 0:
            continue        
        flop_ind = loss_vals.index.searchsorted(flop)
        if flop_ind > 0:
            flop_ind += -1 + np.abs(np.log(flop_vals[flop_ind-1:flop_ind+1]/flop)).argmin()
        rel_err = np.exp(np.abs(np.log(flop_vals[flop_ind]/flop))) - 1
        if rel_err > flop_tolerance:
            continue

        if len(flop_vals) > 1:
            flop_slice = flop_vals[max(0,flop_ind-5):flop_ind+5]
            loss_slide = loss_vals.iloc[max(0,flop_ind-5):flop_ind+5]
            loss_interp = np.exp(np.interp(np.log(flop), np.log(flop_slice), np.log(loss_slide)))
            out.append(dict(n=row[n_key], t=flop / row[flop_per_token_key], loss=loss_interp))
        else:
            out.append(dict(n=row[n_key], t=loss_vals.index[flop_ind] / row[flop_per_token_key], loss=loss_vals.iloc[flop_ind]))
        if keep_bs_lr_keys:
            out[-1].update({k: row[k] for k in [bs_key, 'lr']})

    return pd.DataFrame(out)


def power_law_fit(df, x, y, weighted=False):
    if isinstance(y, (list, tuple)):
        out = {}
        for yy in y:
            if 'loss' not in yy:
                out.update(power_law_fit(df, x, yy, weighted=weighted))
            else:
                df = df.copy()
                out.update(fit_loss_with_saturation(df, weighted=weighted))
        return out
    else:
        X_data = np.log(df.dropna()[x].values).reshape(-1, 1)
        y_data = np.log(df.dropna()[y].values)
        std_key = f'{y}_star_std'
        if weighted and std_key in df.columns:
            y_data_std = df.dropna()[std_key].values
            w = 1 / y_data_std ** 2
        else:
            w = None

        clf = LinearRegression().fit(X_data, y_data, sample_weight=w)
        return {f'{y}_exponent': clf.coef_.item(),
                f'{y}_coef': np.exp(clf.intercept_),
                f'{y}_r2': clf.score(X_data, y_data)}


def fit_compute_optimal_power_laws(optimal_pairs, bootstrap_data, bootstrap_num=None, bootstrap_num_loss=200, fit_loss=True):
    keys_to_fit = ['n', 't', 'multiplier']
    if fit_loss:
        keys_to_fit.append('loss')
    out = {'basic': power_law_fit(optimal_pairs.reset_index(), 'flops', keys_to_fit),
           'weighted': power_law_fit(optimal_pairs.reset_index(), 'flops', keys_to_fit, weighted=True)}
    bootstrap_samples = bootstrap_data.dropna().set_index('flops')[
        ['n_stars', 't_stars', 'multiplier_stars', 'loss_stars', 'n_star_std', 't_star_std', 'loss_star_std']].rename(
        columns=lambda x: x.replace('_stars', ''))
    if bootstrap_num is None:
        bootstrap_num = bootstrap_samples[['n', 't', 'multiplier']].applymap(len).min().min()

    for name, is_weighted in dict(bootstrap=False, bootstrap_weighted=True).items():
        bs_smaples_arr = [
            power_law_fit(bootstrap_samples.applymap(lambda x: maybe_get_item(x, i)).reset_index(),
            'flops', ['loss'], weighted=is_weighted)
            for i in range(bootstrap_num_loss)
            ] if fit_loss else []
        bs_smaples_arr.extend([power_law_fit(
            bootstrap_samples.applymap(lambda x: maybe_get_item(x, i)).reset_index(),
            'flops', ['n', 't', 'multiplier'], weighted=is_weighted)
            for i in range(bootstrap_num)])
        out[name] = bs_smaples_arr
    bootstrap_medians = bootstrap_samples.applymap(np.median)
    out.update({
        'bs_median': power_law_fit(bootstrap_medians.reset_index(), 'flops', keys_to_fit),
        'bs_median_weighted': power_law_fit(bootstrap_medians.reset_index(), 'flops', keys_to_fit, weighted=True)})
    return out


def get_noise_for_loss(loss, bootstrap_iters, noise_low=0.005, noise_high=0.1, l_threshold_high=6, l_threshold_low=3):
    basic_noise = np.random.normal(0, 1, (bootstrap_iters, len(loss) // bootstrap_iters))
    noise_adjusted_losses = np.zeros((bootstrap_iters, len(loss) // bootstrap_iters))

    for i in range(len(loss) // bootstrap_iters):
        if np.log(loss[i]) >= l_threshold_high:
            log_noise = np.log(noise_high)
        elif np.log(loss[i]) <= l_threshold_low:
            log_noise = np.log(noise_low)
        else:
            log_noise = np.interp(np.log(loss[i]), [np.log(l_threshold_low), np.log(l_threshold_high)], [np.log(noise_low), np.log(noise_high)])
        noise_factor = np.exp(log_noise)
        noise_adjusted_losses[:, i] = loss[i] + noise_factor * basic_noise[:, i]
        
    return noise_adjusted_losses.flatten()


def vectorized_interp_with_seed_noise(df, n_interp_, bootstrap_iters, seed_noise=None,
                                      min_std_factor=0.33, tok_or_n='n'):
    if seed_noise is None:
        seed_noise = {}
    interp_num = len(n_interp_)
    stacked_df = pd.concat([df] * bootstrap_iters).reset_index(drop=True)
    stacked_df['loss'] = get_noise_for_loss(stacked_df.loss, bootstrap_iters=bootstrap_iters, **seed_noise)

    batch_ids = np.repeat(np.arange(bootstrap_iters), len(df))
    stacked_df['batch_id'] = batch_ids
    stacked_df.sort_values(by=['batch_id', tok_or_n], inplace=True)

    def batch_interp(batch):
        interp = scipy.interpolate.Akima1DInterpolator(np.log(batch[tok_or_n]), np.log(batch['loss']))
        return np.exp(interp(np.log(n_interp_)))

    interpolated_values = stacked_df.groupby('batch_id').apply(batch_interp)

    # Find the index of the minimum interpolated loss value per batch
    min_indices = interpolated_values.apply(np.argmin)
    results = [n_interp_[idx] if idx != 0 and idx != interp_num - 1 else None for idx in min_indices]
    valid_results_loss = [interpolated_values[i][idx] for i, idx in enumerate(min_indices) if idx != 0 and idx != interp_num - 1]
    # Filter None values and calculate statistics
    valid_results = [result for result in results if result is not None]
    if len(valid_results) < bootstrap_iters // 2:
        return None, 0, None, None, None
    else:
        n_star_std_ = np.std(np.log(valid_results))
        min_std = min_std_factor * np.log(n_interp_[1] / n_interp_[0])  # this assumes a roughly uniform grid
        n_star_std_ = max(n_star_std_, min_std) * (bootstrap_iters / len(valid_results))
        loss_star_std_ = np.std(np.log(valid_results_loss))
        min_std_loss = min_std_factor * min([np.log(df.loss.iloc[i+1] / df.loss.iloc[i]) for i in range(len(df) - 1)])
        loss_star_std_ = max(loss_star_std_, min_std_loss) * (bootstrap_iters / len(valid_results_loss))
        return n_star_std_, None, valid_results, valid_results_loss, loss_star_std_


def interpolation(df_, interp_num, bootstrap_iters, seed_noise, min_std_factor, interp_num_multiplier, std_method, col):
    interp_ = np.geomspace(df_[col].min(), df_[col].max(), interp_num)
    df_ = df_.sort_values(col)
    interpolator = scipy.interpolate.Akima1DInterpolator(np.log(df_[col]), np.log(df_.loss))
    loss_interp_ = np.exp(interpolator(np.log(interp_)))
    star_ind_ = loss_interp_.argmin()

    if std_method == 'add_seed_noise':
        star_std_, _, noised_stars_, noised_loss, loss_star_std = vectorized_interp_with_seed_noise(
            df_, interp_, bootstrap_iters, seed_noise, min_std_factor * interp_num_multiplier, tok_or_n=col)
    else:
        star_std_ = None
        noised_stars_ = []

    return star_ind_, star_std_, noised_stars_, interp_, loss_interp_, noised_loss, loss_star_std


def interp_flop(big_df, loss_key, flop_vals=[8e16, 3e17, 6e17, 3e18, 6e18, 1e19], groupby_action='min',
                warmup_remove_factor=1e-12,
                interp_num_multiplier=25,
                n_key='params', n_star_std_method='add_seed_noise', t_star_std_method='add_seed_noise',
                bootstrap_iters=1000,
                min_std_factor=0.33,
                seed_noise=None, flop_tolerance=0.1,
                flop_per_token_key='flops_per_token',
                bs_median_as_obs=True,
                keep_bs_lr_keys=False,
                ):
    out = []
    optimal_pairs = []
    max_loss, min_loss = 0, 1e12

    for c in flop_vals:
        df_ = fetch_flop(big_df, c, loss_key=loss_key, 
                         warmup_remove_factor=warmup_remove_factor, n_key=n_key, 
                         flop_per_token_key=flop_per_token_key,
                         flop_tolerance=flop_tolerance, keep_bs_lr_keys=keep_bs_lr_keys)

        if len(df_) < 3:
            out.append(dict(n_interp=None, loss_interp=None, t_interp=None, 
                            loss_interp_tok=None, opt_ind=None, opt_tok_ind=None, flops=c))
            continue
        if 'bs' in df_.columns and 'lr' in df_.columns:
            df_sweep_opt_eta = df_.groupby(['n','bs']).apply(minimize_with_interp).drop(['bs', 'n'], axis=1).reset_index()
            df_sweep_opt_eta_and_bs = df_sweep_opt_eta.groupby(['n']).apply(lambda x: minimize_with_interp(x, x_key='bs')).drop('n', axis=1).reset_index()
            df_ = df_sweep_opt_eta_and_bs[['n']]
            df_['t'] 
            print(df_.iloc[0].loss)
        elif groupby_action == 'min':
            df_ = df_.loc[df_.groupby(['n']).loss.idxmin()]
        elif groupby_action == 'mean':
            df_ = df_.groupby('n').mean()
        else:
            raise ValueError(f'Unknown groupby_action {groupby_action}')
        df_ = df_.reset_index()

        interp_num = (len(df_) - 1) * interp_num_multiplier

        max_loss, min_loss = max(max_loss, df_.loss.max()), min(min_loss, df_.loss.min())
        
        n_star_ind_, n_star_std_, noised_n_stars_, n_interp_, loss_interp_, noised_loss, loss_star_std = interpolation(
            df_, interp_num, bootstrap_iters, seed_noise, min_std_factor, interp_num_multiplier, n_star_std_method, 'n')

        t_star_ind_, t_star_std_, noised_t_stars_, t_interp_, loss_interp_tok_, noised_loss, _ = interpolation(
            df_, interp_num, bootstrap_iters, seed_noise, min_std_factor, interp_num_multiplier, t_star_std_method, 't')
        
        if n_star_ind_ != 0 and n_star_ind_ != interp_num -1 and noised_n_stars_ is not None:
            optimal_pairs.append(
                dict(flops=c, n=n_interp_[n_star_ind_], t=t_interp_[t_star_ind_], multiplier=c / 6 / (n_interp_[n_star_ind_]**2),
                     loss=loss_interp_.min(), loss_t=loss_interp_tok_.min(),
                     n_vals=df_.n.values, t_vals=df_.t.values, loss_vals=df_.loss
                    )
            )
        else:
            optimal_pairs.append(
                dict(flops=c, n=None, t=None, loss=None, loss_t=None,
                        n_vals=df_.n.values, t_vals=df_.t.values, loss_vals=df_.loss
                    )
            )
        out.append(
            dict(n_interp=n_interp_, loss_interp=loss_interp_, 
                 t_interp=t_interp_, loss_interp_tok=loss_interp_tok_, 
                 opt_ind=n_star_ind_, opt_tok_ind=t_star_ind_, flops=c, 
                 orig_n=df_.n, orig_t=df_.t, orig_loss=df_.loss)
            )
        if n_star_std_method == 'add_seed_noise':
            out[-1]['n_star_std'] = n_star_std_
            out[-1]['n_stars'] = noised_n_stars_
            out[-1]['multiplier_stars'] = (c / (6 * np.array(noised_n_stars_)**2)) if noised_n_stars_ is not None else None
            optimal_pairs[-1]['n_star_std'] = n_star_std_ 
            
            out[-1]['multiplier_star_std'] = 2 * n_star_std_ if n_star_std_ is not None else None
            optimal_pairs[-1]['multiplier_star_std'] = 2 * n_star_std_ if n_star_std_ is not None else None

            out[-1]['t_star_std'] = t_star_std_
            out[-1]['t_stars'] = noised_t_stars_
            optimal_pairs[-1]['t_star_std'] = t_star_std_

            out[-1]['loss_stars'] = noised_loss
            out[-1]['loss_star_std'] = loss_star_std 
            optimal_pairs[-1]['loss_star_std'] = loss_star_std

    out_df = pd.DataFrame(out)
    optimal_pairs_df = pd.DataFrame(optimal_pairs)

    if bs_median_as_obs:
        for ind, row in optimal_pairs_df.iterrows():
            if row['n'] is None or np.isnan(row['n']):
                continue
            flop = row['flops']
            data_row = out_df.set_index('flops').loc[flop]
            for key in ['n', 't', 'multiplier', 'loss']:
                optimal_pairs_df.at[ind, key] = np.median(data_row[key + '_stars']) if data_row[key + '_stars'] is not None else row[key]

    return out_df, optimal_pairs_df, max_loss, min_loss


def fit_loss_with_saturation(df, weighted=False, fit_min_flop=1e16, fit_max_flop=5e17):
    def model_func(F, a, e, alpha):
        return np.logaddexp(a - alpha * np.log(F), e)
    df = df.dropna().query(f'flops > {fit_min_flop} and flops < {fit_max_flop}')
    loss = df['loss'].values
    flops = df['flops'].values
    
    alpha_vals = np.arange(0, 0.4, 0.1)
    e_vals = np.arange(-1, 1.5, 0.5)
    a_vals = np.arange(0, 30, 5)
    best_loss = np.inf
    best_params = None
    results_dict = {}
    for alpha, e, a in list(product(alpha_vals, e_vals, a_vals)):
        init_params = [a, e, alpha]
        try:
            if weighted:
                popt, _ = curve_fit(model_func, flops, np.log(loss), p0=init_params, sigma=1 / df['loss_star_std'].values ** 2, method='trf', ftol=1e-6, xtol=1e-6, max_nfev=100)
            else:
                popt, _ = curve_fit(model_func, flops, np.log(loss), p0=init_params, method='trf', ftol=1e-6, xtol=1e-6, max_nfev=100)
            result_loss = huber_loss_objective(popt, flops, loss)
            results_dict[tuple(init_params)] = {'params': popt, 'loss': result_loss}
            if result_loss < best_loss:
                best_loss = result_loss
                best_params = popt
        except RuntimeError:
            continue
        
    if best_params is not None:
        A = np.exp(best_params[0])
        E = np.exp(best_params[1])
        alpha = best_params[2]
        return {'A': A, 'E': E, 'alpha': alpha}
    else:
        return None


def predict_and_estimate_cost(df, predict_targets, confidence_level=0.05, anytime=True, start_flop=None,
                              max_models_per_flop=50, max_excess_loss=1.0, max_multiplier=100, base_flop_vals=None, seed_noise=None,
                              **predict_args):
    res = []
    if base_flop_vals is None:
        base_flop_vals = FLOP_VALS
    for i in range(3, len(base_flop_vals) + 1):
        flop_vals = base_flop_vals[:i]
        data, optimal_pairs, max_loss, min_loss = interp_flop(
            df, flop_vals=flop_vals, seed_noise=seed_noise, **predict_args)
        if start_flop is None:
            start_flop = optimal_pairs['flops'].min()
        fit_results = fit_compute_optimal_power_laws(optimal_pairs.query('flops >= @start_flop'), data.query('flops >= @start_flop'), fit_loss=False)
        # extract the prediction
        def extract_single_prediction(fit_dict):
            pred = dict(exponent=fit_dict['n_exponent'], coef=fit_dict['n_coef'])
            for i, target in enumerate(predict_targets):
                pred[f'prediction_at_{target:.3g}'] = fit_dict['n_coef'] * (target ** fit_dict['n_exponent'])
            return pred

        point_prediction = extract_single_prediction(fit_results['bs_median_weighted'])
        bs_predictions = pd.DataFrame([extract_single_prediction(x) for x in fit_results['bootstrap_weighted']])
        confidence_interval = bs_predictions.quantile([confidence_level / 2, 1 - confidence_level / 2])
        confidence_interval.index = ['lo', 'hi']
        confidence_interval_dict = {f'{k}_{q}': v for (q, k), v in confidence_interval.stack().items()}
        # confidence_interval_dict = {(k, q): v for (q, k), v in confidence_interval.stack().items()}

        # compute the cost
        # make a list of model/flops pairs
        relevant_models = []
        for flop, flop_df in optimal_pairs.dropna().explode(['n_vals', 'loss_vals']).groupby('flops'):
            flop_df = flop_df.copy()
            flop_df['excess_loss'] = flop_df['loss_vals'] - flop_df['loss_vals'].min()
            flop_df['multiplier'] = flop / (6 * flop_df.n_vals ** 2)
            flop_df = flop_df.sort_values('excess_loss').query(
                'excess_loss < @max_excess_loss & multiplier <= @max_multiplier').iloc[
                      :max_models_per_flop]
            relevant_models.append(flop_df[['flops', 'n_vals', 'loss_vals']])
        relevant_models_df = pd.concat(relevant_models, axis=0, ignore_index=True)

        if anytime:
            flop_vals = relevant_models_df.groupby('n_vals').flops.max()
        else:
            flop_vals = relevant_models_df.flops
        cost = flop_vals.sum()

        res.append(dict(max_flop=optimal_pairs.dropna().flops.max(),
                        cost=cost, optimal_pairs=optimal_pairs, bs_data=data, fit_results=fit_results,
                        bs_predictions=bs_predictions) | point_prediction | confidence_interval_dict)

    return pd.DataFrame(res).set_index('cost').sort_index(axis=1)


def perform_varying_compute_analysis(df, predict_targets,
                                     config_compute, confidence_level=0.05, start_flop=None,
                                     flop_vals=None, seed=42):
    np.random.seed(seed)

    if flop_vals is None:
        flop_vals = FLOP_VALS
    df = df.copy()
    dataset, hparams, warmup, decay, param_count, val = config_compute
    show_df = df.query("dataset==@dataset and hparams==@hparams and warmup==@warmup and decay==@decay")

    df_compute = predict_and_estimate_cost(
            show_df, predict_targets, base_flop_vals=flop_vals, start_flop=start_flop,  **ISOFLOP_ARGS[config_compute[-2:]], seed_noise=SEED_ARGS[config_compute]
        )
    return pd.DataFrame([dict(dataset=dataset, hparams=hparams, warmup=warmup,decay=decay,param_count=param_count,val=val,
                            base_flop_vals=flop_vals, predict_targets=predict_targets, confidence_level=confidence_level, results_df=df_compute)])


def perform_main_analysis(results_df, configs,
                          flop_vals=None,
                          seed=42, seed_noise_args=None, 
                          fit_loss=True, keep_bs_lr_keys=False
                          ):
    np.random.seed(seed)

    if flop_vals is None:
        flop_vals = FLOP_VALS
    if seed_noise_args is None:
        seed_noise_args = SEED_ARGS
    df = results_df.copy()
    out = []
    for config in configs:
        dataset, hparams, warmup, decay, param_count, val = config
        show_df = df.query(f"dataset=='{dataset}' and hparams=='{hparams}' and warmup=='{warmup}' and decay=='{decay}'")

        if len(show_df) == 0:
            continue
        data, optimal_pairs, max_loss, min_loss = interp_flop(
            show_df, seed_noise = seed_noise_args[config], 
            flop_vals=flop_vals, **ISOFLOP_ARGS[config[-2:]],
            keep_bs_lr_keys=keep_bs_lr_keys,
        )

        fit_results = fit_compute_optimal_power_laws(optimal_pairs, data, fit_loss=fit_loss)

        out.append(dict(dataset=dataset, hparams=hparams, warmup=warmup, decay=decay, param_count=param_count, val=val, 
                        optimal_pairs=optimal_pairs, fit_results=fit_results,
                        data=data, max_loss=max_loss, min_loss=min_loss,))
    return pd.DataFrame(out)


# For hparams sweep - basic interpolation
def minimize_with_interp(df, x_key='lr', y_key='loss', interp_num=100, groupby_action='min', interpolator=scipy.interpolate.Akima1DInterpolator):
    df = df.copy().reset_index()
    if groupby_action == 'min':
        df = df.loc[df.groupby([x_key])[y_key].idxmin()]  
        # take the best value of lr, etc., if there are multiple ones - could potentially do better by interpolating here too
        df = df.set_index(x_key)
    elif groupby_action == 'mean':
        df = df.groupby(x_key).mean()
    else:
        raise ValueError(f'Unknown groupby_action {groupby_action}')
    df = df.sort_index()

    if len(df) < 2:
        return pd.DataFrame({x_key: [np.nan], y_key: [np.nan], 'on_edge': True})
    
    xlog, ylog = np.log(df.index.values), np.log(df[y_key].values)
    interp = interpolator(xlog, ylog)

    xlog_i = np.linspace(xlog.min(), xlog.max(), interp_num)
    ylog_i = interp(xlog_i)

    x_i, y_i = np.exp(xlog_i), np.exp(ylog_i)

    argmin_xlog_i = xlog_i[y_i.argmin()]
    argmin_x_i = x_i[y_i.argmin()]
    on_edge = argmin_x_i < np.exp(xlog[1]) or argmin_x_i > np.exp(xlog[-2])
    out = {x_key: [argmin_x_i], 'on_edge': int(on_edge)+1}

    for key in df.columns:
        if key == 'index':
            continue
        out[key] = [np.exp(interpolator(xlog, np.log(df[key].values))(argmin_xlog_i))]

    if x_key + '_star' in df.columns:
        interp_l = interpolator(xlog, np.log(df[y_key + '_star'].values))
        out['loss_star'] = [np.exp(interp_l(np.log(df[x_key + '_star'].values[0])))]
    
    return pd.DataFrame(out).set_index(x_key)


# For hparams sweep - for the full df
def create_pivot_df(df, loss_col='loss'):
    df = df.copy()
    pivot_df = df.pivot_table(index=['lr', 'bs', 'beta2'], columns='params', values=loss_col, aggfunc='first').reset_index()
    pivot_df.columns = [f'final_loss_smoothed_{col:.2e}' if isinstance(col, float) else col for col in pivot_df.columns]
    
    pivot_df = pivot_df.sort_values(by=['bs', 'lr']).reset_index(drop=True)
    return pivot_df


# For hparams sweep
def get_interpolated_hparams_dfs(df_sweep, min_params_for_fit=2.5e7, max_params_for_fit=1.1e8, lr_star=None, bs_star=None):
    df_sweep = df_sweep.copy()
    if lr_star is not None:
        df_sweep['lr_star'] = lr_star
        df_sweep['bs_star'] = bs_star
    df_sweep_opt_eta = df_sweep.drop('excess_loss', axis=1).groupby(['params','bs']).apply(minimize_with_interp).drop(['bs', 'params'], axis=1).reset_index()
    df_sweep_opt_eta_and_bs = df_sweep_opt_eta.groupby(['params']).apply(lambda x: minimize_with_interp(x, x_key='bs')).drop('params', axis=1).reset_index()
    query_str = f"params > {min_params_for_fit} and params < {max_params_for_fit}"
    fit_dict_bs = power_law_fit(df_sweep_opt_eta_and_bs.query(query_str).reset_index().copy(), 'params', 'bs')
    fit_dict_lr = power_law_fit(df_sweep_opt_eta_and_bs.query(query_str).reset_index().copy(), 'params', 'lr')
    fit = {'bs': fit_dict_bs, 'lr': fit_dict_lr}
    return df_sweep_opt_eta_and_bs, fit


# For seed variance analysis (lower multipliers)
def hparams_other_multipliers(df, multipliers, lr_star=None, bs_star=None):
    df_sweep_extended = df.copy()
    fits = []
    all_M_df = pd.DataFrame()
    cols_to_take = {M: ['params', 'lr', 'bs', 'beta2', f'loss_mul_{M:.2f}', f'excess_loss_{M:.2f}'] for M in multipliers}
    if 'lr_star' in df_sweep_extended.columns:
        for M in cols_to_take.keys():
            cols_to_take[M] += ['lr_star', 'bs_star']
    else:
        lr_star = None
        bs_star = None

    for M in multipliers:
        df_sweep_M = df_sweep_extended[cols_to_take[M]].copy()
        df_sweep_M.rename(columns={f'loss_mul_{M:.2f}': 'loss', f'excess_loss_{M:.2f}': 'excess_loss'}, inplace=True)
        if 'lr_star' in df_sweep_M.columns:
            lr_star = df_sweep_M['lr_star']
            bs_star = df_sweep_M['bs_star']
            df_sweep_M['loss_star'] = df_sweep_M['loss']
        
        df_sweep_opt_eta_and_bs_M, fit_M = get_interpolated_hparams_dfs(df_sweep_M, lr_star=lr_star, bs_star=bs_star)
        df_sweep_opt_eta_and_bs_M['M'] = M
        added_dict = {'M': M}
        added_dict.update({k: v for k, v in fit_M['bs'].items()})
        added_dict.update({k: v for k, v in fit_M['lr'].items()})
        fits.append(added_dict)
        all_M_df = pd.concat([all_M_df, df_sweep_opt_eta_and_bs_M])
    all_M_df['loss_diff'] = all_M_df['loss_star'] - all_M_df['loss']
    all_M_df = all_M_df[all_M_df['loss_diff'] > 0]
    window_size = 10
    def apply_median_filter(group):
        group['loss_diff_smoothed'] = group['loss_diff'].rolling(window=window_size, center=True, min_periods=1).median()
        return group

    all_M_df = all_M_df.groupby('params', group_keys=False).apply(apply_median_filter).reset_index(drop=True)

    return all_M_df, fits


def fit_l_star_vs_N_for_M(df, smoothed=False):
    fits = {}
    loss_key = 'loss_diff' if not smoothed else 'loss_diff_smoothed'
    for M, df_M in df.groupby('M'):
        interpolator = scipy.interpolate.Akima1DInterpolator(np.log(df_M['params']), np.log(df_M[loss_key]))
        fits[M] = interpolator
    return fits


def preform_analysis_with_sweep_data(summary_df, sweep_fit):
    df = summary_df.copy()
    data = df.iloc[-1]['data']
    data_sweep = data[['flops']].copy()
    loss_interp = [None] * len(data_sweep)
    loss_orig = [None] * len(data_sweep)
    optimal_n = [None] * len(data_sweep)
    optimal_n_std = [None] * len(data_sweep)
    n_interp_list = [None] * len(data_sweep)
    p_est_ = [None] * len(data_sweep)
    bs_list = []
    for i, c in enumerate(data.flops.unique()):
        data_c = data.loc[data.flops == c].iloc[0]
        mask_fit_orig = ((c / (data_c['orig_n']**2) / 6) >= 2) & ((c / (data_c['orig_n']**2) / 6) <= 30)
        def closest_k(n_):
            multiplier = c / (6*n_**2)
            if multiplier > 20:
                multiplier = 40 - multiplier
            return min(sweep_fit.keys(), key=lambda x: abs(x - multiplier))
        loss_orig_delta = [np.exp(sweep_fit[closest_k(n_)](np.log(n_))) for n_ in data_c['orig_n']]
        loss_orig_delta = data_c['orig_loss'][mask_fit_orig] - np.array(loss_orig_delta)[mask_fit_orig]
        df_ = pd.DataFrame({'n': data_c['orig_n'][mask_fit_orig], 'loss': loss_orig_delta[mask_fit_orig]})
        df_ = df_.dropna()
        if len(df_) <= 2:
            continue
        star_ind_, star_std_, noised_stars_, interp_, loss_interp_, _, _ = interpolation(df_, (len(df_) - 1) * 100, 1000, RW_SEED_CONFIG, 0.33, 100, 'add_seed_noise', 'n')
        if star_std_ is not None and i > 0:
            bs_list.append((c, noised_stars_))
        optimal_n_std[i] = star_std_
        n_interp_list[i] = interp_
        loss_interp[i] = loss_interp_
        loss_orig[i] = loss_orig_delta
        optimal_n[i] = (interp_[star_ind_], loss_interp_[star_ind_]) if star_std_ is not None else None
        if star_std_ is not None:
            p_est_[i] = (c, interp_[star_ind_], star_std_)

    bootstrap_samples = []
    min_len = min([len(bs_list[i][1]) for i in range(1, len(bs_list))])
    bootstrap_samples = [pd.DataFrame([(bs_list[j][0], bs_list[j][1][i]) for j in range(len(bs_list))], columns=['c', 'n']) for i in range(min_len)]
    fit_results_bootstrap = [power_law_fit(bootstrap_samples[i], 'c', 'n') for i in range(min_len)]
    df = pd.DataFrame([(c, np.median(n_stars)) for (c, n_stars) in bs_list if (c, n_stars) is not None], columns=['c', 'n'])
    df['n_star_std'] = [std_ for std_ in optimal_n_std if std_ is not None][1:]
    fit_results = power_law_fit(df, 'c', 'n', weighted=True)
   
    data_sweep['loss_interp'] = loss_interp
    data_sweep['loss_orig'] = loss_orig
    data_sweep['optimal_n'] = optimal_n
    data_sweep['n_interp'] = n_interp_list
    return data_sweep, pd.DataFrame(p_est_, columns=['c', 'n', 'n_star_std']), {'bs_median_weighted': fit_results, 'bs': fit_results_bootstrap}, 


# For seed variance analysis
def perform_seed_var_analysis(seed_df):
    seed_df = seed_df.copy()
    groupby_keys = ['params', 'dataset']
    agg_cols = ['train/lr', 'train/loss', 'train/loss_smoothed', 'val/loss', 'val/loss_std']
    results = []
    for groupby_vals, df_ in seed_df.groupby(groupby_keys):
        row = dict(zip(groupby_keys, groupby_vals))
        for col in agg_cols:
            row[col + '_concat'] = pd.concat([v.to_frame() for v in df_[col]], axis=1).dropna()
            for op in ('mean', 'std'):
                row[col + '_' + op] = row[col + '_concat'].agg(op, axis=1)
        results.append(row)
    return pd.DataFrame(results)