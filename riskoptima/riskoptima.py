###############################################################################
#                                riskoptima.py
###############################################################################

"""
Author: Jordi Corbilla

Date: 16/02/2025

This module (extended) provides various financial functions and tools for analyzing 
and handling portfolio data learned from EDHEC Business School, computing statistical 
metrics, and optimizing portfolios based on different criteria. 

Main features include:
- Loading and formatting financial datasets (Fama-French, EDHEC Hedge Fund Index, etc.)
- Computing portfolio statistics (returns, volatility, Sharpe ratio, etc.)
- Running backtests on different portfolio strategies
- Efficient Frontier plotting
- Value at Risk (VaR) and Conditional Value at Risk (CVaR) computations
- Portfolio optimization based on different risk metrics
- Mean Variance Optimization
- Machine learning strategies (Linear Regression, XGBoost, SVR, etc.)
- Black-Litterman adjusted returns
- Market correlation and financial ratios
- Monte Carlo-based portfolio analysis and probability distributions
- Black-Scholes option pricing model
- Heston stochastic volatility model for option pricing
- Merton Jump Diffusion model for option pricing
- Heatmap visualization of option prices using Monte Carlo simulation
- Comprehensive bond analytics including duration, convexity, and yield sensitivity

Dependencies: 
    pandas, numpy, scipy, statsmodels, yfinance, datetime, scikit-learn,
    matplotlib, seaborn, xgboost, squarify
"""

import pandas as pd
import numpy as np
import scipy.stats as si
import statsmodels.api as sm
import math
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from xgboost import XGBRegressor
from sklearn.svm import SVR
from datetime import date, datetime, timedelta
import seaborn as sns
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.dates import DateFormatter
from matplotlib.dates import AutoDateLocator
import matplotlib.patches as patches
import squarify
import matplotlib as mpl
import matplotlib.ticker as mticker
import os

import warnings
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*DataFrame.std with axis=None is deprecated.*"
)

class RiskOptima:
    TRADING_DAYS = 260  # default is 260, though 252 is also common
    VERSION = '1.25.0'

    @staticmethod
    def get_trading_days():
        """
        Returns the number of trading days for a given year, by default 260
        """
        return RiskOptima.TRADING_DAYS

    @staticmethod
    def download_data_yfinance(assets, start_date, end_date):
        """
        Downloads the adjusted close price data from Yahoo Finance for the given assets
        between the specified date range.

        :param assets: List of asset tickers.
        :param start_date: Start date for data in 'YYYY-MM-DD' format.
        :param end_date: End date for data in 'YYYY-MM-DD' format.
        :return: A pandas DataFrame of adjusted close prices.
        """
        data = yf.download(assets, start=start_date, end=end_date, progress=False)
        return data['Close']

    @staticmethod
    def get_ffme_returns(file_path):
        """
        Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
        """
        me_m = pd.read_csv(file_path, header=0, index_col=0, na_values=-99.99)
        returns = me_m[['Lo 10', 'Hi 10']]
        returns.columns = ['SmallCap', 'LargeCap']
        returns = returns / 100
        returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
        return returns

    @staticmethod
    def get_fff_returns(file_path):
        """
        Load the Fama-French Research Factor Monthly Dataset
        """
        returns = pd.read_csv(file_path, header=0, index_col=0, na_values=-99.99) / 100
        returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
        return returns

    @staticmethod
    def get_hfi_returns(file_path):
        """
        Load and format the EDHEC Hedge Fund Index Returns
        """
        hfi = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
        hfi = hfi / 100
        hfi.index = hfi.index.to_period('M')
        return hfi

    @staticmethod
    def get_ind_file(file_path, filetype, weighting="vw", n_inds=30):
        """
        Load and format the Ken French Industry Portfolios files
        Variant is a tuple of (weighting, size) where:
            weighting is one of "ew", "vw"
            number of inds is 30 or 49
        """
        if filetype == "returns":
            divisor = 100
        elif filetype == "nfirms":
            divisor = 1
        elif filetype == "size":
            divisor = 1
        else:
            raise ValueError("filetype must be one of: returns, nfirms, size")

        ind = pd.read_csv(file_path, header=0, index_col=0, na_values=-99.99) / divisor
        ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
        ind.columns = ind.columns.str.strip()
        return ind

    @staticmethod
    def get_ind_returns(file_path, weighting="vw", n_inds=30):
        """
        Load and format the Ken French Industry Portfolios Monthly Returns
        """
        return RiskOptima.get_ind_file(file_path, "returns", weighting=weighting, n_inds=n_inds)

    @staticmethod
    def get_ind_nfirms(file_path, n_inds=30):
        """
        Load and format the Ken French 30 Industry Portfolios Average number of Firms
        """
        return RiskOptima.get_ind_file(file_path, "nfirms", n_inds=n_inds)

    @staticmethod
    def get_ind_size(file_path, n_inds=30):
        """
        Load and format the Ken French 30 Industry Portfolios Average size (market cap)
        """
        return RiskOptima.get_ind_file(file_path, "size", n_inds=n_inds)

    @staticmethod
    def get_ind_market_caps(nfirms_file_path, size_file_path, n_inds=30, weights=False):
        """
        Load the industry portfolio data and derive the market caps
        """
        ind_nfirms = RiskOptima.get_ind_nfirms(nfirms_file_path, n_inds=n_inds)
        ind_size = RiskOptima.get_ind_size(size_file_path, n_inds=n_inds)
        ind_mktcap = ind_nfirms * ind_size
        if weights:
            total_mktcap = ind_mktcap.sum(axis=1)
            ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
            return ind_capweight
        return ind_mktcap

    @staticmethod
    def get_total_market_index_returns(nfirms_file_path, size_file_path, returns_file_path, n_inds=30):
        """
        Load the 30 industry portfolio data and derive the returns of a capweighted total market index
        """
        ind_capweight = RiskOptima.get_ind_market_caps(nfirms_file_path, size_file_path, n_inds=n_inds)
        ind_return = RiskOptima.get_ind_returns(returns_file_path, weighting="vw", n_inds=n_inds)
        total_market_return = (ind_capweight * ind_return).sum(axis="columns")
        return total_market_return

    @staticmethod
    def skewness(returns):
        """
        Alternative to scipy.stats.skew()
        Computes the skewness of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_returns = returns - returns.mean()
        sigma_returns = returns.std(ddof=0)
        exp = (demeaned_returns**3).mean()
        return exp / sigma_returns**3

    @staticmethod
    def kurtosis(returns):
        """
        Alternative to scipy.stats.kurtosis()
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_returns = returns - returns.mean()
        sigma_returns = returns.std(ddof=0)
        exp = (demeaned_returns**4).mean()
        return exp / sigma_returns**4

    @staticmethod
    def compound(returns):
        """
        Returns the result of compounding the set of returns
        """
        return np.expm1(np.log1p(returns).sum())

    @staticmethod
    def annualize_returns(returns, periods_per_year):
        """
        Annualizes a set of returns
        """
        compounded_growth = (1 + returns).prod()
        n_periods = returns.shape[0]
        return compounded_growth**(periods_per_year / n_periods) - 1

    @staticmethod
    def annualize_volatility(returns, periods_per_year):
        """
        Annualizes the volatility of a set of returns
        """
        return returns.std(axis=0) * (periods_per_year**0.5)

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate, periods_per_year=None):
        """
        Calculate the Sharpe Ratio for a given set of investment returns.

        :param returns: pandas Series or numpy array of investment returns.
        :param float risk_free_rate: Annualized risk-free rate (e.g., yield on government bonds).
        :param int periods_per_year: Number of periods per year (e.g., 12 for monthly, 252 for daily). 
                                     Defaults to RiskOptima.get_trading_days() for daily data.
        :return: float Sharpe Ratio.
        """
        if periods_per_year is None:
            periods_per_year = RiskOptima.get_trading_days()

        rf_per_period = (1 + risk_free_rate)**(1 / periods_per_year) - 1
        excess_returns = returns - rf_per_period

        ann_excess_returns = RiskOptima.annualize_returns(excess_returns, periods_per_year)
        ann_volatility = RiskOptima.annualize_volatility(returns, periods_per_year)

        return ann_excess_returns / ann_volatility

    @staticmethod
    def is_normal(returns, level=0.01):
        """
        Applies the Jarque-Bera test to determine if a Series is normal or not
        Test is applied at the 1% level by default
        Returns True if the hypothesis of normality is accepted, False otherwise
        """
        if isinstance(returns, pd.DataFrame):
            return returns.aggregate(RiskOptima.is_normal)
        else:
            statistic, p_value = si.jarque_bera(returns)
            return p_value > level

    @staticmethod
    def drawdown(return_series: pd.Series):
        """
        Takes a time series of asset returns.
        Returns a DataFrame with columns for
        the wealth index, 
        the previous peaks, and 
        the percentage drawdown
        """
        wealth_index = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame({
            "Wealth": wealth_index, 
            "Previous Peak": previous_peaks, 
            "Drawdown": drawdowns
        })

    @staticmethod
    def semideviation(returns):
        """
        Returns the semideviation aka negative semideviation of returns
        returns must be a Series or a DataFrame, else raises a TypeError
        """
        if isinstance(returns, pd.Series):
            is_negative = returns < 0
            return returns[is_negative].std(ddof=0)
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(RiskOptima.semideviation)
        else:
            raise TypeError("Expected returns to be a Series or DataFrame")

    @staticmethod
    def var_historic(returns, level=5):
        """
        Returns the historic Value at Risk at a specified level
        i.e. returns the number such that "level" percent of the returns
        fall below that number, and the (100-level) percent are above
        """
        if isinstance(returns, pd.DataFrame):
            return returns.aggregate(RiskOptima.var_historic, level=level)
        elif isinstance(returns, pd.Series):
            return -np.percentile(returns, level)
        else:
            raise TypeError("Expected returns to be a Series or DataFrame")

    @staticmethod
    def cvar_historic(returns, level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(returns, pd.Series):
            is_beyond = returns <= -RiskOptima.var_historic(returns, level=level)
            return -returns[is_beyond].mean()
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(RiskOptima.cvar_historic, level=level)
        else:
            raise TypeError("Expected returns to be a Series or DataFrame")

    @staticmethod
    def var_gaussian(returns, level=5, modified=False):
        """
        Returns the Parametric Gaussian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        z = norm.ppf(level / 100)
        if modified:
            s = RiskOptima.skewness(returns)
            k = RiskOptima.kurtosis(returns)
            z = (z +
                 (z**2 - 1) * s / 6 +
                 (z**3 - 3 * z) * (k - 3) / 24 -
                 (2 * z**3 - 5 * z) * (s**2) / 36)
        return -(returns.mean() + z * returns.std(ddof=0))

    @staticmethod
    def portfolio_return(weights, returns):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ returns

    @staticmethod
    def portfolio_volatility(weights, covmat):
        """
        Computes the volatility of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        volatility = (weights.T @ covmat @ weights)**0.5
        return volatility

    @staticmethod
    def plot_ef2(n_points, expected_returns, cov, style):
        """
        Plots the 2-asset efficient frontier
        """
        if expected_returns.shape[0] != 2:
            raise ValueError("plot_ef2 can only plot 2-asset frontiers")
        weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
        rets = [RiskOptima.portfolio_return(w, expected_returns) for w in weights]
        volatilities = [RiskOptima.portfolio_volatility(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": volatilities
        })
        return ef.plot.line(x="Volatility", y="Returns", style=style)

    @staticmethod
    def minimize_volatility(target_return, expected_returns, cov):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix
        """
        n = expected_returns.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        return_is_target = {'type': 'eq', 'args': (expected_returns,),
                            'fun': lambda weights, expected_returns: target_return - RiskOptima.portfolio_return(weights, expected_returns)}
        weights = minimize(RiskOptima.portfolio_volatility, init_guess, args=(cov,),
                           method='SLSQP', options={'disp': False},
                           constraints=(weights_sum_to_1, return_is_target),
                           bounds=bounds)
        return weights.x

    @staticmethod
    def tracking_error(returns_a, returns_b):
        """
        Returns the Tracking Error between the two return series
        """
        return np.sqrt(((returns_a - returns_b)**2).sum())

    @staticmethod
    def max_sharpe_ratio(riskfree_rate, expected_returns, cov):
        """
        Returns the weights of the portfolio that gives you the maximum Sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        n = expected_returns.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

        def neg_sharpe(weights, riskfree_rate, expected_returns, cov):
            r = RiskOptima.portfolio_return(weights, expected_returns)
            vol = RiskOptima.portfolio_volatility(weights, cov)
            return -(r - riskfree_rate) / vol

        weights = minimize(neg_sharpe, init_guess, args=(riskfree_rate, expected_returns, cov),
                           method='SLSQP', options={'disp': False},
                           constraints=(weights_sum_to_1,), bounds=bounds)
        return weights.x

    @staticmethod
    def global_minimum_volatility(cov):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        given a covariance matrix
        """
        n = cov.shape[0]
        return RiskOptima.max_sharpe_ratio(0, np.repeat(1, n), cov)

    @staticmethod
    def optimal_weights(n_points, expected_returns, cov):
        """
        Returns a list of weights that represent a grid of n_points on the efficient frontier
        """
        target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
        weights = [RiskOptima.minimize_volatility(target_return, expected_returns, cov) 
                   for target_return in target_returns]
        return weights

    @staticmethod
    def plot_ef(n_points, expected_returns, cov, style='.-',
                legend=False, show_cml=False, riskfree_rate=0, 
                show_ew=False, show_gmv=False):
        """
        Plots the multi-asset efficient frontier
        """
        weights = RiskOptima.optimal_weights(n_points, expected_returns, cov)
        rets = [RiskOptima.portfolio_return(w, expected_returns) for w in weights]
        volatilities = [RiskOptima.portfolio_volatility(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": volatilities
        })
        ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
        if show_cml:
            ax.set_xlim(left=0)
            w_msr = RiskOptima.max_sharpe_ratio(riskfree_rate, expected_returns, cov)
            r_msr = RiskOptima.portfolio_return(w_msr, expected_returns)
            vol_msr = RiskOptima.portfolio_volatility(w_msr, cov)
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed',
                    linewidth=2, markersize=10)
        if show_ew:
            n = expected_returns.shape[0]
            w_ew = np.repeat(1 / n, n)
            r_ew = RiskOptima.portfolio_return(w_ew, expected_returns)
            vol_ew = RiskOptima.portfolio_volatility(w_ew, cov)
            ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
        if show_gmv:
            w_gmv = RiskOptima.global_minimum_volatility(cov)
            r_gmv = RiskOptima.portfolio_return(w_gmv, expected_returns)
            vol_gmv = RiskOptima.portfolio_volatility(w_gmv, cov)
            ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        return ax

    @staticmethod
    def plot_ef_ax(n_points, expected_returns, cov, style='.-',
                   legend=False, show_cml=False, riskfree_rate=0,
                   show_ew=False, show_gmv=False, ax=None):
        """
        Plots the multi-asset efficient frontier
        """
        weights = RiskOptima.optimal_weights(n_points, expected_returns, cov)
        rets = [RiskOptima.portfolio_return(w, expected_returns) for w in weights]
        volatilities = [RiskOptima.portfolio_volatility(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": volatilities
        })
        ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend, ax=ax)
        w_msr = None
        w_gmv = None
        if show_cml:
            ax.set_xlim(left=0)
            w_msr = RiskOptima.max_sharpe_ratio(riskfree_rate, expected_returns, cov)
            r_msr = RiskOptima.portfolio_return(w_msr, expected_returns)
            vol_msr = RiskOptima.portfolio_volatility(w_msr, cov)
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.plot(cml_x, cml_y, color='blue', marker='o', linestyle='dashed', 
                    linewidth=2, markersize=10, label='Capital Market Line (CML)')
        if show_ew:
            n = expected_returns.shape[0]
            w_ew = np.repeat(1 / n, n)
            r_ew = RiskOptima.portfolio_return(w_ew, expected_returns)
            vol_ew = RiskOptima.portfolio_volatility(w_ew, cov)
            ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10,
                    label='Naive portfolio (EWP)')
        if show_gmv:
            w_gmv = RiskOptima.global_minimum_volatility(cov)
            r_gmv = RiskOptima.portfolio_return(w_gmv, expected_returns)
            vol_gmv = RiskOptima.portfolio_volatility(w_gmv, cov)
            ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10,
                    label='Global Minimum-variance Portfolio (GMV)')
        return ax, w_msr, w_gmv

    @staticmethod
    def run_cppi(risky_returns, safe_returns=None, m=3, start=1000, floor=0.8,
                 riskfree_rate=0.03, drawdown=None):
        """
        Run a backtest of the CPPI strategy, given a set of returns for the risky asset
        Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
        """
        dates = risky_returns.index
        n_steps = len(dates)
        account_value = start
        floor_value = start * floor
        peak = account_value
        if isinstance(risky_returns, pd.Series): 
            risky_returns = pd.DataFrame(risky_returns, columns=["R"])

        if safe_returns is None:
            safe_returns = pd.DataFrame().reindex_like(risky_returns)
            safe_returns.values[:] = riskfree_rate / 12

        account_history = pd.DataFrame().reindex_like(risky_returns)
        risky_w_history = pd.DataFrame().reindex_like(risky_returns)
        cushion_history = pd.DataFrame().reindex_like(risky_returns)
        floorval_history = pd.DataFrame().reindex_like(risky_returns)
        peak_history = pd.DataFrame().reindex_like(risky_returns)

        for step in range(n_steps):
            if drawdown is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak * (1 - drawdown)
            cushion = (account_value - floor_value) / account_value
            risky_w = m * cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1 - risky_w
            risky_alloc = account_value * risky_w
            safe_alloc = account_value * safe_w
            account_value = risky_alloc * (1 + risky_returns.iloc[step]) + safe_alloc * (1 + safe_returns.iloc[step])
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w
            account_history.iloc[step] = account_value
            floorval_history.iloc[step] = floor_value
            peak_history.iloc[step] = peak

        risky_wealth = start * (1 + risky_returns).cumprod()
        backtest_result = {
            "Wealth": account_history,
            "Risky Wealth": risky_wealth, 
            "Risk Budget": cushion_history,
            "Risky Allocation": risky_w_history,
            "m": m,
            "start": start,
            "floor": floor,
            "risky_returns": risky_returns,
            "safe_returns": safe_returns,
            "drawdown": drawdown,
            "peak": peak_history,
            "floorval_history": floorval_history
        }
        return backtest_result

    @staticmethod
    def summary_stats(returns, riskfree_rate=0.03):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of returns
        """
        ann_returns = returns.aggregate(RiskOptima.annualize_returns, periods_per_year=12)
        ann_volatility = returns.aggregate(RiskOptima.annualize_volatility, periods_per_year=12)
        ann_sr = returns.aggregate(RiskOptima.sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
        dd = returns.aggregate(lambda returns: RiskOptima.drawdown(returns).Drawdown.min())
        skew = returns.aggregate(RiskOptima.skewness)
        kurt = returns.aggregate(RiskOptima.kurtosis)
        cf_var5 = returns.aggregate(RiskOptima.var_gaussian, modified=True)
        hist_cvar5 = returns.aggregate(RiskOptima.cvar_historic)
        return pd.DataFrame({
            "Annualized Return": ann_returns,
            "Annualized Volatility": ann_volatility,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Cornish-Fisher VaR (5%)": cf_var5,
            "Historic CVaR (5%)": hist_cvar5,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown": dd
        })

    @staticmethod
    def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15,
            steps_per_year=12, s_0=100.0, prices=True):
        """
        Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
        :param n_years:  The number of years to generate data for
        :param n_paths: The number of scenarios/trajectories
        :param mu: Annualized Drift, e.g. Market Return
        :param sigma: Annualized Volatility
        :param steps_per_year: granularity of the simulation
        :param s_0: initial value
        :return: a numpy array of n_paths columns and n_years*steps_per_year rows
        """
        dt = 1 / steps_per_year
        n_steps = int(n_years * steps_per_year) + 1
        rets_plus_1 = np.random.normal(loc=(1 + mu)**dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))
        rets_plus_1[0] = 1
        ret_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
        return ret_val

    @staticmethod
    def regress(dependent_variable, explanatory_variables, alpha=True):
        """
        Runs a linear regression to decompose the dependent variable into the explanatory variables
        returns an object of type statsmodel's RegressionResults on which you can call
           .summary() to print a full summary
           .params for the coefficients
           .tvalues and .pvalues for the significance levels
           .rsquared_adj and .rsquared for quality of fit
        """
        if alpha:
            explanatory_variables = explanatory_variables.copy()
            explanatory_variables["Alpha"] = 1
        
        lm = sm.OLS(dependent_variable, explanatory_variables).fit()
        return lm

    @staticmethod
    def portfolio_tracking_error(weights, ref_returns, bb_returns):
        """
        Returns the tracking error between the reference returns
        and a portfolio of building block returns held with given weights
        """
        return RiskOptima.tracking_error(ref_returns, (weights * bb_returns).sum(axis=1))

    @staticmethod
    def style_analysis(dependent_variable, explanatory_variables):
        """
        Returns the optimal weights that minimizes the tracking error between
        a portfolio of the explanatory variables and the dependent variable
        """
        n = explanatory_variables.shape[1]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        solution = minimize(RiskOptima.portfolio_tracking_error,
                            init_guess, 
                            args=(dependent_variable, explanatory_variables,),
                            method='SLSQP', options={'disp': False},
                            constraints=(weights_sum_to_1,),
                            bounds=bounds)
        weights = pd.Series(solution.x, index=explanatory_variables.columns)
        return weights

    @staticmethod
    def ff_analysis(returns, factors):
        """
        Returns the loadings of returns on the Fama French Factors
        which can be read in using get_fff_returns()
        the index of returns must be a (not necessarily proper) subset of the index of factors
        returns is either a Series or a DataFrame
        """
        if isinstance(returns, pd.Series):
            dependent_variable = returns
            explanatory_variables = factors.loc[returns.index]
            tilts = RiskOptima.regress(dependent_variable, explanatory_variables).params
        elif isinstance(returns, pd.DataFrame):
            tilts = pd.DataFrame({col: RiskOptima.ff_analysis(returns[col], factors) for col in returns.columns})
        else:
            raise TypeError("returns must be a Series or a DataFrame")
        return tilts

    @staticmethod
    def weight_ew(returns, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
        """
        Returns the weights of the EW portfolio based on the asset returns "returns" as a DataFrame
        If supplied a set of capweights and a capweight tether, it is applied and reweighted 
        """
        n = len(returns.columns)
        ew = pd.Series(1 / n, index=returns.columns)
        if cap_weights is not None:
            cw = cap_weights.loc[returns.index[0]]
            if microcap_threshold is not None and microcap_threshold > 0:
                microcap = cw < microcap_threshold
                ew[microcap] = 0
                ew = ew / ew.sum()
            if max_cw_mult is not None and max_cw_mult > 0:
                ew = np.minimum(ew, cw * max_cw_mult)
                ew = ew / ew.sum()
        return ew

    @staticmethod
    def weight_cw(returns, cap_weights, **kwargs):
        """
        Returns the weights of the CW portfolio based on the time series of capweights
        """
        w = cap_weights.loc[returns.index[0]]
        return w / w.sum()

    @staticmethod
    def backtest_ws(returns, estimation_window=60, weighting=weight_ew, verbose=False, **kwargs):
        """
        Backtests a given weighting scheme, given some parameters:
        returns : asset returns to use to build the portfolio
        estimation_window: the window to use to estimate parameters
        weighting: the weighting scheme to use, must be a function that takes "returns", and a variable number of keyword-value arguments
        """
        n_periods = returns.shape[0]
        windows = [(start, start + estimation_window) for start in range(n_periods - estimation_window)]
        weights = [weighting(returns.iloc[win[0]:win[1]], **kwargs) for win in windows]
        weights = pd.DataFrame(weights, index=returns.iloc[estimation_window:].index, columns=returns.columns)
        portfolio_returns = (weights * returns).sum(axis="columns", min_count=1)
        return portfolio_returns

    @staticmethod
    def sample_covariance(returns, **kwargs):
        """
        Returns the sample covariance of the supplied returns
        """
        return returns.cov()

    @staticmethod
    def weight_gmv(returns, cov_estimator=sample_covariance, **kwargs):
        """
        Produces the weights of the GMV portfolio given a covariance matrix of the returns 
        """
        est_cov = cov_estimator(returns, **kwargs)
        return RiskOptima.global_minimum_volatility(est_cov)

    @staticmethod
    def cc_covariance(returns, **kwargs):
        """
        Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
        """
        rhos = returns.corr()
        n = rhos.shape[0]
        rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
        ccor = np.full_like(rhos, rho_bar)
        np.fill_diagonal(ccor, 1.)
        sd = returns.std(axis=0)
        return pd.DataFrame(ccor * np.outer(sd, sd), index=returns.columns, columns=returns.columns)
    
    @staticmethod
    def shrinkage_covariance(returns, delta=0.5, **kwargs):
        """
        Covariance estimator that shrinks between the 
        Sample Covariance and the Constant Correlation Estimators
        """
        prior = RiskOptima.cc_covariance(returns, **kwargs)
        sample = RiskOptima.sample_covariance(returns, **kwargs)
        return delta * prior + (1 - delta) * sample

    @staticmethod
    def risk_contribution(weights, cov):
        """
        Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
        """
        total_portfolio_var = RiskOptima.portfolio_volatility(weights, cov)**2
        marginal_contrib = cov @ weights
        risk_contrib = np.multiply(marginal_contrib, weights.T) / total_portfolio_var
        return risk_contrib

    @staticmethod
    def target_risk_contributions(target_risk, cov):
        """
        Returns the weights of the portfolio that gives you the weights such
        that the contributions to portfolio risk are as close as possible to
        the target_risk, given the covariance matrix
        """
        n = cov.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

        def msd_risk(weights, target_risk, cov):
            w_contribs = RiskOptima.risk_contribution(weights, cov)
            return ((w_contribs - target_risk)**2).sum()

        weights = minimize(msd_risk, init_guess, args=(target_risk, cov),
                           method='SLSQP', options={'disp': False},
                           constraints=(weights_sum_to_1,), bounds=bounds)
        return weights.x

    @staticmethod
    def equal_risk_contributions(cov):
        """
        Returns the weights of the portfolio that equalizes the contributions
        of the constituents based on the given covariance matrix
        """
        n = cov.shape[0]
        return RiskOptima.target_risk_contributions(target_risk=np.repeat(1 / n, n), cov=cov)

    @staticmethod
    def weight_erc(returns, cov_estimator=sample_covariance, **kwargs):
        """
        Produces the weights of the ERC portfolio given a covariance matrix of the returns 
        """
        est_cov = cov_estimator(returns, **kwargs)
        return RiskOptima.equal_risk_contributions(est_cov)

    @staticmethod
    def discount(t, r):
        """
        Compute the price of a pure discount bond that pays a dollar at time period t
        and r is the per-period interest rate
        returns a |t| x |r| Series or DataFrame
        r can be a float, Series or DataFrame
        returns a DataFrame indexed by t
        """
        discounts = pd.DataFrame([(r+1)**-i for i in t])
        discounts.index = t
        return discounts

    @staticmethod
    def pv(flows, r):
        """
        Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
        r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
        """
        dates = flows.index
        discounts = RiskOptima.discount(dates, r)
        return discounts.multiply(flows, axis='rows').sum()

    @staticmethod
    def funding_ratio(assets, liabilities, r):
        """
        Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
        """
        return RiskOptima.pv(assets, r)/RiskOptima.pv(liabilities, r)

    @staticmethod
    def inst_to_ann(r):
        """
        Convert an instantaneous interest rate to an annual interest rate
        """
        return np.expm1(r)

    @staticmethod
    def ann_to_inst(r):
        """
        Convert an instantaneous interest rate to an annual interest rate
        """
        return np.log1p(r)

    @staticmethod
    def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
        """
        Generate random interest rate evolution over time using the CIR model
        b and r_0 are assumed to be the annualized rates, not the short rate
        and the returned values are the annualized rates as well
        """
        if r_0 is None: 
            r_0 = b
        r_0 = RiskOptima.ann_to_inst(r_0)
        dt = 1/steps_per_year
        num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
        
        shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
        rates = np.empty_like(shock)
        rates[0] = r_0
    
        ## For Price Generation
        h = math.sqrt(a**2 + 2*sigma**2)
        prices = np.empty_like(shock)

        def price(ttm, r):
            _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
            _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
            _P = _A*np.exp(-_B*r)
            return _P

        prices[0] = price(n_years, r_0)

        for step in range(1, num_steps):
            r_t = rates[step-1]
            d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
            rates[step] = abs(r_t + d_r_t)
            # generate prices at time t as well ...
            prices[step] = price(n_years-step*dt, rates[step])

        rates = pd.DataFrame(data=RiskOptima.inst_to_ann(rates), index=range(num_steps))
        prices = pd.DataFrame(data=prices, index=range(num_steps))
        return rates, prices

    @staticmethod
    def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
        """
        Returns the series of cash flows generated by a bond,
        indexed by the payment/coupon number
        """
        n_coupons = round(maturity*coupons_per_year)
        coupon_amt = principal*coupon_rate/coupons_per_year
        coupon_times = np.arange(1, n_coupons+1)
        cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
        cash_flows.iloc[-1] += principal
        return cash_flows

    @staticmethod
    def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
        """
        Computes the price of a bond that pays regular coupons until maturity
        at which time the principal and the final coupon is returned
        This is not designed to be efficient, rather,
        it is to illustrate the underlying principle behind bond pricing!
        If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
        and the bond value is computed over time.
        i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
        """
        if isinstance(discount_rate, pd.DataFrame):
            pricing_dates = discount_rate.index
            prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
            for t in pricing_dates:
                prices.loc[t] = RiskOptima.bond_price(maturity-t/coupons_per_year,
                                                      principal, coupon_rate,
                                                      coupons_per_year,
                                                      discount_rate.loc[t])
            return prices
        else:
            if maturity <= 0:
                return principal+principal*coupon_rate/coupons_per_year
            cash_flows = RiskOptima.bond_cash_flows(maturity, principal,
                                                    coupon_rate, coupons_per_year)
            return RiskOptima.pv(cash_flows, discount_rate/coupons_per_year)

    @staticmethod
    def macaulay_duration(flows, discount_rate):
        """
        Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
        """
        discounted_flows = RiskOptima.discount(flows.index, discount_rate)*flows
        weights = discounted_flows/discounted_flows.sum()
        return np.average(flows.index, weights=weights)

    @staticmethod
    def match_durations(cf_t, cf_s, cf_l, discount_rate):
        """
        Returns the weight W in cf_s that, along with (1-W) in cf_l 
        will have an effective duration that matches cf_t
        """
        d_t = RiskOptima.macaulay_duration(cf_t, discount_rate)
        d_s = RiskOptima.macaulay_duration(cf_s, discount_rate)
        d_l = RiskOptima.macaulay_duration(cf_l, discount_rate)
        return (d_l - d_t)/(d_l - d_s)

    @staticmethod
    def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
        """
        Computes the total return of a Bond based on monthly bond prices and coupon payments
        Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
        and that dividends are reinvested in the bond
        """
        coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
        t_max = monthly_prices.index.max()
        pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
        coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
        total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
        return total_returns.dropna()

    @staticmethod
    def bt_mix(r1, r2, allocator, **kwargs):
        """
        Runs a back test (simulation) of allocating between a two sets of returns
        r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
        allocator is a function that takes two sets of returns and allocator specific parameters, and produces
        an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
        Returns a T x N DataFrame of the resulting N portfolio scenarios
        """
        if not r1.shape == r2.shape:
            raise ValueError("r1 and r2 should have the same shape")
        weights = allocator(r1, r2, **kwargs)
        if not weights.shape == r1.shape:
            raise ValueError("Allocator returned weights with a different shape than the returns")
        r_mix = weights*r1 + (1-weights)*r2
        return r_mix

    @staticmethod
    def fixedmix_allocator(r1, r2, w1, **kwargs):
        """
        Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
        PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
         each column is a scenario
         each row is the price for a timestep
        Returns an T x N DataFrame of PSP Weights
        """
        return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)
    
    @staticmethod
    def terminal_values(rets):
        """
        Computes the terminal values from a set of returns supplied as a T x N DataFrame
        Return a Series of length N indexed by the columns of rets
        """
        return (rets+1).prod()

    @staticmethod
    def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
        """
        Produce Summary Statistics on the terminal values per invested dollar
        across a range of N scenarios
        rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
        Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
        """
        terminal_wealth = (rets+1).prod()
        breach = terminal_wealth < floor
        reach = terminal_wealth >= cap
        p_breach = breach.mean() if breach.sum() > 0 else np.nan
        p_reach = breach.mean() if reach.sum() > 0 else np.nan
        e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
        e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
        sum_stats = pd.DataFrame.from_dict({
            "mean": terminal_wealth.mean(),
            "std" : terminal_wealth.std(axis=0),
            "p_breach": p_breach,
            "e_short":e_short,
            "p_reach": p_reach,
            "e_surplus": e_surplus
        }, orient="index", columns=[name])
        return sum_stats

    @staticmethod
    def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
        """
        Allocates weights to r1 starting at start_glide and ends at end_glide
        by gradually moving from start_glide to end_glide over time
        """
        n_points = r1.shape[0]
        n_col = r1.shape[1]
        path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
        paths = pd.concat([path]*n_col, axis=1)
        paths.index = r1.index
        paths.columns = r1.columns
        return paths

    @staticmethod
    def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
        """
        Allocate between PSP and GHP with the goal to provide exposure to the upside
        of the PSP without going violating the floor.
        Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
        of the cushion in the PSP
        Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
        """
        if zc_prices.shape != psp_r.shape:
            raise ValueError("PSP and ZC Prices must have the same shape")
        n_steps, n_scenarios = psp_r.shape
        account_value = np.repeat(1, n_scenarios)
        floor_value = np.repeat(1, n_scenarios)
        w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
        for step in range(n_steps):
            floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
            cushion = (account_value - floor_value)/account_value
            psp_w = (m*cushion).clip(0, 1) # same as applying min and max
            ghp_w = 1-psp_w
            psp_alloc = account_value*psp_w
            ghp_alloc = account_value*ghp_w
            # recompute the new account value at the end of this step
            account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
            w_history.iloc[step] = psp_w
        return w_history

    @staticmethod
    def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
        """
        Allocate between PSP and GHP with the goal to provide exposure to the upside
        of the PSP without going violating the floor.
        Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
        of the cushion in the PSP
        Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
        """
        n_steps, n_scenarios = psp_r.shape
        account_value = np.repeat(1, n_scenarios)
        floor_value = np.repeat(1, n_scenarios)
        ### For MaxDD
        peak_value = np.repeat(1, n_scenarios)
        w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
        for step in range(n_steps):
            ### For MaxDD
            floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
            cushion = (account_value - floor_value)/account_value
            psp_w = (m*cushion).clip(0, 1) # same as applying min and max
            ghp_w = 1-psp_w
            psp_alloc = account_value*psp_w
            ghp_alloc = account_value*ghp_w
            # recompute the new account value at the end of this step
            account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
            ### For MaxDD
            peak_value = np.maximum(peak_value, account_value) ### For MaxDD
            w_history.iloc[step] = psp_w
        return w_history

    @staticmethod
    def discount_v2(t, r, freq):
        """
        Compute the price of a pure discount bond that pays a dollar at time period t
        and r is the per-period interest rate
        returns a DataFrame indexed by t
        """
        discounts = pd.DataFrame([(1 + r / freq) ** -(t * freq) for t in t], 
                                 index=t, columns=['df'])
        return discounts

    @staticmethod
    def bond_cash_flows_v2(n_periods, par, coupon_rate, freq):
        """
        Generate bond cash flows
        """
        coupon = par * coupon_rate / freq
        cash_flows = np.full(n_periods, coupon)
        cash_flows[-1] += par
        return cash_flows

    @staticmethod
    def bond_price_v2(cash_flows, yield_rate, freq):
        """
        Calculate the price of the bond
        """
        n = len(cash_flows)
        times = np.arange(1, n + 1) / freq
        discount_factors = RiskOptima.discount(times, yield_rate).values.flatten()
        present_values = cash_flows * discount_factors
        return sum(present_values)

    @staticmethod
    def macaulay_duration_v2(cash_flows, yield_rate, freq):
        """
        Calculate the Macaulay Duration
        """
        n = len(cash_flows)
        times = np.arange(1, n + 1) / freq
        discount_factors = RiskOptima.discount(times, yield_rate).values.flatten()
        present_values = cash_flows * discount_factors

        weighted_sum = sum(times * present_values)
        total_present_value = sum(present_values)

        return weighted_sum / total_present_value

    @staticmethod
    def macaulay_duration_v3(cash_flows, yield_rate, freq):
        """
        Calculate the Macaulay Duration, Modified Duration, Dollar Duration, PVBP, and Convexity.
        """
        n = len(cash_flows)
        times = np.arange(1, n + 1) / freq
        discount_factors = RiskOptima.discount_v2(times, yield_rate, freq).values.flatten()
        present_values = cash_flows * discount_factors
        total_present_value = sum(present_values)
        weights = present_values / total_present_value
        weighted_average_times = times * weights

        # Compute Macaulay Duration
        macaulay_duration = sum(weighted_average_times)

        # Compute Modified Duration
        modified_duration = macaulay_duration / (1 + yield_rate / freq)

        # Compute Dollar Duration
        bond_price = total_present_value
        face_value = 1000  # Assuming face value is 1000
        dollar_duration = modified_duration * bond_price / face_value

        # Compute PVBP (DV01)
        pvbp = dollar_duration / 100

        # Compute Convexity
        convexity_factors = times * (times + 1)
        convexity = sum(present_values * convexity_factors) / (bond_price * (1 + yield_rate / freq) ** 2)

        # Store detailed calculation breakdown
        df = pd.DataFrame({
            't': times,
            'df': discount_factors,
            'cf': cash_flows,
            'pv': present_values,
            'weight': weights,
            'wat': weighted_average_times
        })

        totals = pd.DataFrame({
            't': ['Total'],
            'df': [''],
            'cf': [sum(cash_flows)],
            'pv': [total_present_value],
            'weight': [sum(weights)],
            'wat': [sum(weighted_average_times)]
        })

        df = pd.concat([df, totals], ignore_index=True)

        # Store bond measures in a separate results DataFrame
        bond_measures = pd.DataFrame({
            "Macaulay Duration": [macaulay_duration],
            "Modified Duration": [modified_duration],
            "Dollar Duration": [dollar_duration],
            "PVBP (DV01)": [pvbp],
            "Convexity": [convexity]
        })

        return df, bond_measures

    @staticmethod
    def calculate_statistics(data, risk_free_rate=0.0):
        """
        Calculates daily returns, covariance matrix, mean daily returns, 
        annualized returns, annualized volatility, and Sharpe ratio 
        for the entire dataset.
    
        :param data: A pandas DataFrame of adjusted close prices.
        :param risk_free_rate: The risk-free rate, default is 0.0 (for simplicity).
        :return: daily_returns (DataFrame), cov_matrix (DataFrame)
        """
        daily_returns = data.pct_change(fill_method=None).dropna()
        cov_matrix = daily_returns.cov()
        return daily_returns, cov_matrix

    @staticmethod
    def run_monte_carlo_simulation(daily_returns, cov_matrix, 
                                   num_portfolios=100_000, risk_free_rate=0.0):
        """
        Runs the Monte Carlo simulation to generate a large number of random portfolios,
        calculates their performance metrics (annualized return, volatility, Sharpe ratio),
        and returns a DataFrame of results as well as an array of the weight vectors.
    
        :param daily_returns: DataFrame of asset daily returns.
        :param cov_matrix: Covariance matrix of asset daily returns.
        :param num_portfolios: Number of random portfolios to simulate.
        :param risk_free_rate: Risk-free rate to be used in Sharpe ratio calculation.
        :return: (simulated_portfolios, weights_record)
        """
        results = np.zeros((4, num_portfolios))
        weights_record = np.zeros((len(daily_returns.columns), num_portfolios))

        for i in range(num_portfolios):
            weights = np.random.random(len(daily_returns.columns))
            weights /= np.sum(weights)
            weights_record[:, i] = weights

            portfolio_return = np.sum(weights * daily_returns.mean()) * RiskOptima.TRADING_DAYS
    
            portfolio_stddev = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights))
            ) * np.sqrt(RiskOptima.TRADING_DAYS)
    
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
            results[0, i] = portfolio_return
            results[1, i] = portfolio_stddev
            results[2, i] = sharpe_ratio
            results[3, i] = i

        columns = ['Return', 'Volatility', 'Sharpe Ratio', 'Simulation']
        simulated_portfolios = pd.DataFrame(results.T, columns=columns)
        return simulated_portfolios, weights_record

    @staticmethod
    def get_market_statistics(market_ticker, start_date, end_date, risk_free_rate=0.0):
        """
        Downloads data for a market index (e.g., SPY), then calculates its
        annualized return, annualized volatility, and Sharpe ratio.
        """
        market_data = yf.download([market_ticker], start=start_date, end=end_date, progress=False)['Close']
        if isinstance(market_data, pd.DataFrame):
            market_data = market_data[market_ticker] 
        market_daily_returns = market_data.pct_change(fill_method=None).dropna()

        market_return = market_daily_returns.mean() * RiskOptima.TRADING_DAYS
        market_volatility = market_daily_returns.std(axis=0) * np.sqrt(RiskOptima.TRADING_DAYS)
        market_sharpe_ratio = (market_return - risk_free_rate) / market_volatility

        if hasattr(market_return, 'iloc'):
            market_return = market_return.iloc[0]
        if hasattr(market_volatility, 'iloc'):
            market_volatility = market_volatility.iloc[0]
        if hasattr(market_sharpe_ratio, 'iloc'):
            market_sharpe_ratio = market_sharpe_ratio.iloc[0]

        return market_return, market_volatility, market_sharpe_ratio

    @staticmethod
    def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=252):
        """
        Given weights, return annualized portfolio return and volatility.
        """
        returns = np.sum(mean_returns * weights) * trading_days
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
        return returns, volatility

    @staticmethod
    def min_volatility(weights, mean_returns, cov_matrix):
        """
        Objective function: we want to minimize volatility.
        """
        return RiskOptima.portfolio_performance(weights, mean_returns, cov_matrix)[1]

    @staticmethod
    def efficient_frontier(mean_returns, cov_matrix, num_points=50):
        """
        Calculates the Efficient Frontier by iterating over possible target returns
        and finding the portfolio with minimum volatility for each target return.
        Returns arrays of frontier volatilities, returns, and the corresponding weights.
        """
        results = []
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
        num_assets = len(mean_returns)
        init_guess = num_assets * [1. / num_assets,]
        bounds = tuple((0,1) for _ in range(num_assets))

        for ret in target_returns:
            constraints = (
                {'type':'eq', 'fun': lambda w: np.sum(w) - 1}, 
                {'type':'eq', 'fun': lambda w: RiskOptima.portfolio_performance(w, mean_returns, cov_matrix)[0] - ret}
            )
            result = minimize(RiskOptima.min_volatility, 
                              init_guess, 
                              args=(mean_returns, cov_matrix),
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)
            if result.success:
                vol = RiskOptima.portfolio_performance(result.x, mean_returns, cov_matrix)[1]
                results.append((vol, ret, result.x))

        results = sorted(results, key=lambda x: x[0])
        frontier_volatility = [res[0] for res in results]
        frontier_returns = [res[1] for res in results]
        frontier_weights = [res[2] for res in results]
        return frontier_volatility, frontier_returns, frontier_weights

    @staticmethod
    def get_previous_year_date(end_date, years):
        """
        Calculates the start date by subtracting a given number of years from the end_date,
        ensuring that the start date is a working day (Monday-Friday).

        Args:
            end_date (str): The end date in 'YYYY-MM-DD' format.
            years (int): The number of years to go back.

        Returns:
            str: The calculated start date in 'YYYY-MM-DD' format, adjusted to a working day.
        """
        end_date_obj = date.fromisoformat(end_date)
        start_date = end_date_obj.replace(year=end_date_obj.year - years)

        # Adjust to the most recent working day if it falls on a weekend
        if start_date.weekday() == 5:  # Saturday
            start_date -= timedelta(days=1)
        elif start_date.weekday() == 6:  # Sunday
            start_date -= timedelta(days=2)

        return start_date.strftime('%Y-%m-%d')

    @staticmethod
    def get_previous_working_day():
        """
        Returns the most recent weekday date in 'YYYY-MM-DD' format.
        If today is Monday-Friday, returns today.
        If today is Saturday, returns Friday.
        If today is Sunday, returns Friday.
        """
        today = date.today()
        # Monday=0, Tuesday=1, ..., Saturday=5, Sunday=6
        if today.weekday() == 5:      # Saturday
            today -= timedelta(days=1)
        elif today.weekday() == 6:    # Sunday
            today -= timedelta(days=2)
        return today.strftime('%Y-%m-%d')

    @staticmethod
    def calculate_portfolio_allocation(investment_allocation):
        """
        Normalize portfolio allocations based on the investment amounts provided.
    
        :param dict investment_allocation: A dictionary mapping stock tickers to their investment amounts (e.g., {'AAPL': 1000, 'MSFT': 2000}).
        :return: List of stock tickers and a numpy array of normalized weights.
        """
        total_investment = sum(investment_allocation.values())
        normalized_weights = np.array([amount / total_investment for amount in investment_allocation.values()])
        tickers = list(investment_allocation.keys())
        return tickers, normalized_weights

    @staticmethod
    def fetch_historical_stock_prices(tickers, start_date, end_date):
        """
        Retrieve historical stock price data for a list of tickers using Yahoo Finance.
    
        :param list tickers: List of stock ticker symbols.
        :param str start_date: Start date for historical data in 'YYYY-MM-DD' format.
        :param str end_date: End date for historical data in 'YYYY-MM-DD' format.
        :return: pandas DataFrame containing the adjusted closing prices for the specified tickers.
        """
        stock_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        return stock_data

    @staticmethod
    def perform_mean_variance_optimization(tickers, start_date, end_date,
                                           max_acceptable_volatility,
                                           predefined_returns=None,
                                           min_allocation=0.01,
                                           max_allocation=0.35,
                                           num_simulations=100000):
        """
        Execute mean-variance optimization using Monte Carlo simulation with weight constraints.
    
        :param list tickers: List of stock ticker symbols to optimize.
        :param str start_date: Start date for the historical data in 'YYYY-MM-DD' format.
        :param str end_date: End date for the historical data in 'YYYY-MM-DD' format.
        :param float max_acceptable_volatility: Maximum allowable annualized volatility for the portfolio.
        :param ndarray predefined_returns: (Optional) Predefined annualized returns for the tickers.
        :param float min_allocation: Minimum weight allocation for each stock.
        :param float max_allocation: Maximum weight allocation for each stock.
        :param int num_simulations: Number of Monte Carlo simulations to run.
        :return: Optimal portfolio weights as a numpy array.
        """
        price_data = RiskOptima.fetch_historical_stock_prices(tickers, start_date, end_date)['Close']
        if price_data.empty:
            raise ValueError("No historical data retrieved. Verify the tickers and date range.")

        daily_returns = price_data.pct_change(fill_method=None).dropna()

        if predefined_returns is None:
            predefined_returns = daily_returns.mean() * RiskOptima.TRADING_DAYS

        covariance_matrix = daily_returns.cov() * RiskOptima.TRADING_DAYS
        simulation_results = np.zeros((4, num_simulations))
        weight_matrix = np.zeros((len(tickers), num_simulations))

        for i in range(num_simulations):
            random_weights = np.random.uniform(min_allocation, max_allocation, len(tickers))
            random_weights /= np.sum(random_weights)

            weight_matrix[:, i] = random_weights

            portfolio_return = np.sum(random_weights * predefined_returns)
            portfolio_volatility = np.sqrt(np.dot(random_weights.T, np.dot(covariance_matrix, random_weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

            simulation_results[:, i] = [portfolio_return, portfolio_volatility, sharpe_ratio, i]

        result_columns = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Simulation Index']
        simulation_results_df = pd.DataFrame(simulation_results.T, columns=result_columns)

        feasible_portfolios = simulation_results_df[simulation_results_df['Annualized Volatility'] <= max_acceptable_volatility]
        if feasible_portfolios.empty:
            raise ValueError("No portfolio satisfies the maximum volatility constraint.")

        optimal_index = feasible_portfolios['Sharpe Ratio'].idxmax()
        return weight_matrix[:, int(optimal_index)]

    @staticmethod
    def add_features(stock_prices):
        """
        Add technical indicators like moving averages to the stock data.
        :param stock_prices: DataFrame of stock prices.
        :return: DataFrame with additional feature columns.
        """
        features = pd.DataFrame(stock_prices)
        features['5_day_avg'] = stock_prices.rolling(window=5).mean()
        features['10_day_avg'] = stock_prices.rolling(window=10).mean()
        features['Close'] = stock_prices
        return features

    @staticmethod
    def create_lagged_features(data, lag_days=5):
        """
        Create lagged features for machine learning models.
        :param data: DataFrame containing the stock prices.
        :param lag_days: Number of lag days to include.
        :return: DataFrame with lagged features and target variable.
        """
        lagged_data = data.copy()
        for lag in range(1, lag_days + 1):
            lagged_data[f'lag_{lag}'] = lagged_data['Close'].shift(lag)
        lagged_data.dropna(inplace=True)
        return lagged_data

    @staticmethod
    def evaluate_model(model, X, y):
        """
        Evaluate the model using cross-validation and calculate the average performance metrics.
        :param model: The machine learning model to evaluate.
        :param X: Feature matrix.
        :param y: Target variable.
        :return: Cross-validation score and mean squared error.
        """
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        model.fit(X, y)
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        return np.mean(cv_scores), mse

    @staticmethod
    def predict_with_model(model, feature_data):
        """
        Predict stock returns using the trained model.
        :param model: Trained machine learning model.
        :param feature_data: DataFrame of features for prediction.
        :return: Predicted stock return.
        """
        scaler = StandardScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
        predictions = model.predict(feature_data_scaled)
        return predictions[-1]  # Return the last prediction as the future return
    
    
    @staticmethod
    def generate_stock_predictions(ticker, start_date, end_date, model_type='Linear Regression'):
        """
        Generate stock return predictions and model confidence using a specified model type.
        :param ticker: Stock ticker symbol.
        :param start_date: Start date for the historical data (YYYY-MM-DD).
        :param end_date: End date for the historical data (YYYY-MM-DD).
        :param model_type: Choice of machine learning model ('Linear Regression', 'Random Forest', 'Gradient Boosting').
        :return: Tuple of predicted return and model confidence.
        """
        stock_prices = RiskOptima.download_data_yfinance(ticker, start_date, end_date)
        enriched_data = RiskOptima.add_features(stock_prices)
        prepared_data = RiskOptima.create_lagged_features(enriched_data)

        X = prepared_data.drop('Close', axis=1)
        y = prepared_data['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)

        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        if model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'XGBoost':
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
        elif model_type == 'SVR':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            raise ValueError("Invalid model type. Choose from 'Linear Regression', 'Random Forest', or 'Gradient Boosting'.")
    
        # Train and evaluate the model
        avg_cv_score, mse = RiskOptima.evaluate_model(model, X_train, y_train)
        predicted_return = RiskOptima.predict_with_model(model, X_test)
        return predicted_return, avg_cv_score

    @staticmethod
    def black_litterman_adjust_returns(market_returns, investor_views, view_confidences,
                                       historical_prices, tau=0.025):
        """
        Adjust market returns based on investor views and their confidences using the Black-Litterman model.
    
        :param dict market_returns: Expected market returns for each asset.
        :param dict investor_views: Investor's views on the expected returns of assets.
        :param dict view_confidences: Confidence levels for each investor view.
        :param pandas.DataFrame historical_prices: Historical price data for calculating covariance matrix.
        :param float tau: Market equilibrium uncertainty factor (default 0.025).
        :return: Numpy array of adjusted returns for each asset.
        """
        num_assets = len(market_returns)
        proportion_matrix = np.eye(num_assets)
        views_vector = np.array(list(investor_views.values())).reshape(-1, 1)
        covariance_matrix = historical_prices['Close'].pct_change(fill_method=None).dropna().cov()
        omega_matrix = np.diag([tau / confidence for confidence in view_confidences.values()])

        inv_tau_cov = np.linalg.inv(tau * covariance_matrix)
        inv_omega = np.linalg.inv(omega_matrix)

        adjusted_returns = np.linalg.inv(inv_tau_cov + proportion_matrix.T @ inv_omega @ proportion_matrix)
        adjusted_returns = adjusted_returns @ (
            inv_tau_cov @ np.array(list(market_returns.values())).reshape(-1, 1) +
            proportion_matrix.T @ inv_omega @ views_vector
        )
        return adjusted_returns.flatten()

    @staticmethod
    def compute_market_returns(market_capitalizations, market_index_return):
        """
        Calculate market returns for individual assets based on market capitalizations and index return.
    
        :param dict market_capitalizations: Market capitalizations of assets.
        :param float market_index_return: Return of the overall market index.
        :return: Dictionary mapping tickers to their computed market returns.
        """
        total_market_cap = sum(market_capitalizations.values())
        return {
            ticker: (cap / total_market_cap) * market_index_return
            for ticker, cap in market_capitalizations.items()
        }

    @staticmethod
    def sortino_ratio(returns, risk_free_rate):
        """
        Calculate the Sortino Ratio for a set of investment returns.
    
        :param returns: pandas Series or numpy array of investment returns.
        :param float risk_free_rate: Annualized risk-free rate (e.g., yield on government bonds).
        :return: float Sortino Ratio (returns 0 if downside risk is zero).
        """
        trading_days = RiskOptima.get_trading_days()
        excess_returns = returns - (risk_free_rate / trading_days)
        downside_returns = np.minimum(excess_returns, 0)
        annualized_excess_return = np.mean(excess_returns) * trading_days
        annualized_downside_std_dev = np.std(downside_returns) * np.sqrt(trading_days)
    
        # Return 0 if downside standard deviation is zero
        if (annualized_downside_std_dev == 0).all():  # Use `.all()` for Series comparison
            return 0.0
        return annualized_excess_return / annualized_downside_std_dev

    @staticmethod
    def information_ratio(returns, benchmark_returns):
        """
        Calculate the Information Ratio for a set of investment returns against a benchmark.
    
        :param returns: pandas Series or numpy array of portfolio returns.
        :param benchmark_returns: pandas Series or numpy array of benchmark returns.
        :return: float Information Ratio.
        """
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] > 1:
                raise ValueError("`returns` must be a pandas Series or 1D numpy array, not a DataFrame with multiple columns.")
            returns = returns.squeeze()

        if isinstance(benchmark_returns, pd.DataFrame):
            if benchmark_returns.shape[1] > 1:
                raise ValueError("`benchmark_returns` must be a pandas Series or 1D numpy array, not a DataFrame with multiple columns.")
            benchmark_returns = benchmark_returns.squeeze()

        common_index = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        trading_days = RiskOptima.get_trading_days()

        active_returns = returns - benchmark_returns
    
        if np.allclose(active_returns, 0):  # If active returns are effectively zero
            return 0.0  # Explicitly return 0 for identical returns
    
        annualized_active_return = np.mean(active_returns) * trading_days
        tracking_error = np.std(active_returns) * np.sqrt(trading_days)
    
        if tracking_error == 0:  # Avoid division by zero
            return 0.0
    
        # Return single statistic
        return annualized_active_return / tracking_error

    @staticmethod
    def correlation_with_market(portfolio_returns, market_returns):
        """
        Calculate the correlation between portfolio returns and market index returns.

        :param portfolio_returns: pandas Series of portfolio returns.
        :param market_returns: pandas Series of market index returns.
        :return: float Correlation coefficient.
        """
        common_dates = portfolio_returns.index.intersection(market_returns.index)
        portfolio_aligned = portfolio_returns.loc[common_dates]
        market_aligned = market_returns.loc[common_dates]
        return portfolio_aligned.corr(market_aligned)

    @staticmethod
    def add_table_to_plot(ax, dataframe, column_descriptions=None, column_colors=None, x=1.15, y=0.2, fontsize=8, column_width=0.50):
        """
        Adds a table to the plot with consistent row heights and optional column colors.
        
        :param ax: The matplotlib Axes object.
        :param dataframe: The pandas DataFrame to display as a table.
        :param column_descriptions: Optional list of column header descriptions to override the defaults.
        :param column_colors: List of colors for the table columns. Must match the number of columns in the dataframe.
        :param x: The x-position of the table in Axes coordinates.
        :param y: The y-position of the table in Axes coordinates.
        :param fontsize: Font size for the table text.
        """
        dataframe_reset = dataframe.reset_index()

        if column_descriptions is not None:
            dataframe_reset.columns = column_descriptions

        num_rows = len(dataframe_reset) + 1
        row_height = 0.040  # Fixed height per row (adjust as needed)
        table_height = num_rows * row_height

        table_data = [dataframe_reset.columns.to_list()] + dataframe_reset.values.tolist()

        table = ax.table(
            cellText=table_data,
            colLabels=None,
            colLoc="center",
            loc="right",
            bbox=[x, y, column_width, table_height],  # [left, bottom, width, height] with dynamic height
            cellLoc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.auto_set_column_width(col=list(range(len(dataframe_reset.columns))))

        header_color = "#f2f2f2"  # Light gray background
        for col_index in range(len(dataframe_reset.columns)):
            cell = table[(0, col_index)]
            cell.set_text_props(weight="bold")
            cell.set_facecolor(header_color)
            cell.set_edgecolor("black")

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                continue
            cell.set_edgecolor("black")
            cell.set_linewidth(0.5)
            if column_colors and col < len(column_colors):
                cell.set_facecolor(column_colors[col])

        return table

    @staticmethod
    def consolidate_stats_to_dataframe(titles, stats_lists):
        """
        Converts multiple sets of statistics into a single DataFrame for table display.

        :param titles: A list of titles for the columns (e.g., ["ML & MV Optimized Portfolio", "MV Optimized Portfolio"]).
        :param stats_lists: A list of stats lists, where each stats_list corresponds to a title.
                            Example: [["Sharpe Ratio: 1.95", ...], ["Sharpe Ratio: 1.50", ...]]
        :return: A pandas DataFrame with metrics as rows and titles as columns.
        """
        metrics = [stat.split(":")[0].strip() for stat in stats_lists[0]]
        columns = {}
        for title, stats_list in zip(titles, stats_lists):
            values = [stat.split(":")[1].strip() for stat in stats_list]
            columns[title] = values
        df = pd.DataFrame(columns, index=metrics)
        df.index.name = "Metric"
        return df

    @staticmethod
    def add_portfolio_terms_explanation(ax, x=0.02, y=0.02, fontsize=10):
        """
        Adds an explanation for portfolio-related terms to the chart.

        :param ax: The matplotlib Axes object where the explanation will be added.
        :param x: The x-coordinate of the text box in Axes coordinates (default: 0.02).
        :param y: The y-coordinate of the text box in Axes coordinates (default: 0.02).
        :param fontsize: Font size for the text (default: 10).
        """
        explanation_text = (
            "Portfolio Terms Explanation:\n"
            "1. Return: The expected gain or loss from an investment. Higher is better.\n"
            "2. Volatility: A measure of risk based on price fluctuations. Lower is safer.\n"
            "3. Sharpe Ratio: Measures risk-adjusted return using total volatility. Higher is better.\n"
            "4. Risk-Free Rate: The theoretical return of an investment with zero risk.\n"
            "5. Capital Market Line (CML): Shows risk-return combinations of efficient portfolios.\n"
            "6. Global Minimum Variance Portfolio (GMV): Portfolio with the lowest possible volatility.\n"
            "7. Optimal Portfolio: Portfolio with the best risk-return trade-off based on Sharpe Ratio.\n"
            "8. Naive Portfolio (EWP): Equal-weighted portfolio, used as a baseline for comparison."
        )
        ax.text(
            x, y,
            explanation_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'),
            ha='left'
        )

    @staticmethod
    def plot_efficient_frontier_monte_carlo(
                                asset_table, 
                                start_date='2020-01-01',
                                end_date='2023-01-01',    
                                risk_free_rate=0.0, 
                                num_portfolios=10000,
                                market_benchmark='SPY',
                                set_ticks=False,
                                x_pos_table=1.15,
                                y_pos_table=0.52):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        assets = asset_table['Asset'].tolist()
        current_weights = asset_table['Weight'].values if 'Weight' in asset_table.columns else None
        current_labels = asset_table['Label'].values if 'Label' in asset_table.columns else assets
        
        # Download market data and save it to a CSV (optional)
        asset_data = RiskOptima.download_data_yfinance(assets, start_date, end_date)
        
        data_folder = "data"
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        
        data_path = os.path.join(data_folder, f'market_data_{timestamp}.csv')

        asset_data.to_csv(data_path)
        
        # Compute daily returns and covariance matrix
        daily_returns, cov_matrix = RiskOptima.calculate_statistics(asset_data, risk_free_rate)
        
        # Run Monte Carlo simulation
        simulated_portfolios, weights_record = RiskOptima.run_monte_carlo_simulation(
            daily_returns, cov_matrix,
            num_portfolios=num_portfolios,
            risk_free_rate=risk_free_rate
        )
        
        # Retrieve market benchmark statistics
        market_return, market_volatility, market_sharpe = RiskOptima.get_market_statistics(
            market_benchmark, start_date, end_date, risk_free_rate
        )
        """
        Plot an efficient frontier with additional details
        """
        if set_ticks:
            x_ticks = np.linspace(0, 0.15, 16)  # Adjust the range and number of ticks as needed
            y_ticks = np.linspace(0, 0.30, 16)  # Adjust the range and number of ticks as needed
        
        fig, ax = plt.subplots(figsize=(23, 10))
        fig.subplots_adjust(right=0.80)

        if set_ticks:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1f}%'.format(x * 100)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}%'.format(y * 100)))

        sc = ax.scatter(
            simulated_portfolios['Volatility'],
            simulated_portfolios['Return'],
            c=simulated_portfolios['Sharpe Ratio'],
            cmap='plasma',
            alpha=0.5,
            label='Simulated Portfolios'
        )

        fig.colorbar(sc, ax=ax, label='Sharpe Ratio')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Return')
        
        title=f'[RiskOptima] Efficient Frontier - Monte Carlo Simulation {start_date} to {end_date}'
        
        ax.set_title(title)

        ax.scatter(
            market_volatility, market_return,
            color='red', marker='o', s=100,
            label='Market Benchmark (S&P 500)'
        )

        optimal_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
        optimal_portfolio = simulated_portfolios.loc[optimal_idx]
        optimal_weights = weights_record[:, optimal_idx]

        annual_returns = daily_returns.mean() * RiskOptima.get_trading_days()
        annual_cov = daily_returns.cov() * RiskOptima.get_trading_days()

        n_points = 50
        show_cml = True
        show_ew = True
        show_gmv = True

        _, w_msr, w_gmv = RiskOptima.plot_ef_ax(
            n_points=n_points,
            expected_returns=annual_returns,
            cov=annual_cov,
            style='.-',
            legend=False,
            show_cml=show_cml,
            riskfree_rate=risk_free_rate,
            show_ew=show_ew,
            show_gmv=show_gmv,
            ax=ax
        )

        ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.4, color='lightgray', alpha=0.5)
        ax.set_axisbelow(True)

        if current_weights is not None:
            curr_portfolio_return = np.sum(current_weights * daily_returns.mean()) * RiskOptima.get_trading_days()
            curr_portfolio_vol = np.sqrt(
                np.dot(current_weights.T, np.dot(daily_returns.cov(), current_weights))
            ) * np.sqrt(RiskOptima.get_trading_days())
            current_sharpe = (curr_portfolio_return - risk_free_rate) / curr_portfolio_vol

            ax.scatter(
                curr_portfolio_vol,
                curr_portfolio_return,
                color='black',
                marker='s',
                s=150,
                label='My Current Portfolio'
            )

        ax.scatter(
            optimal_portfolio['Volatility'],
            optimal_portfolio['Return'],
            color='green', marker='*', s=200,
            label='Optimal Portfolio'
        )

        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.08),
            fancybox=True,
            shadow=True,
            ncol=3
        )

        portfolio_df = pd.DataFrame({
            "Security": current_labels,
            "Current\nPortfolio Weights": current_weights,
            "Optimal\nPortfolio Weights": optimal_weights,
            "GMV\nPortfolio Weights": w_gmv,
            "CML\nPortfolio Weights": w_msr
        })
        portfolio_df.set_index("Security", inplace=True)
        portfolio_df = portfolio_df.apply(lambda col: col.map(lambda x: f"{x * 100:.2f}%"))

        RiskOptima.add_table_to_plot(ax, portfolio_df, x=x_pos_table, y=y_pos_table, column_width=0.70, fontsize=9)

        titles = [
            "My Current\nPortfolio",
            "Optimized\nOptimized Portfolio",
            "Market Benchmark\n(S&P 500)"
        ]
        stats_lists = [
            [
                f"Return: {curr_portfolio_return*100:.2f}%",
                f"Volatility: {curr_portfolio_vol*100:.2f}%",
                f"Sharpe Ratio: {current_sharpe:.2f}",
                f"Risk Free Rate: {risk_free_rate*100:.2f}%"
            ],
            [
                f"Return: {optimal_portfolio['Return']*100:.2f}%",
                f"Volatility: {optimal_portfolio['Volatility']*100:.2f}%",
                f"Sharpe Ratio: {optimal_portfolio['Sharpe Ratio']:.2f}",
                f"Risk Free Rate: {risk_free_rate*100:.2f}%"
            ],
            [
                f"Return: {market_return*100:.2f}%",
                f"Volatility: {market_volatility*100:.2f}%",
                f"Sharpe Ratio: {market_sharpe:.2f}",
                f"Risk Free Rate: {risk_free_rate*100:.2f}%"
            ]
        ]
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        stats_df = RiskOptima.consolidate_stats_to_dataframe(titles, stats_lists)
        RiskOptima.add_table_to_plot(ax, stats_df, None, None, x=x_pos_table, y=0.30, column_width=0.70, fontsize=9)
        RiskOptima.add_portfolio_terms_explanation(ax, x=x_pos_table, y=0.00, fontsize=10)

        plots_folder = "plots"
        
        plt.text(
            0.995, -0.20, f"Created by RiskOptima v{RiskOptima.VERSION}",
            fontsize=12, color='gray', alpha=0.7, transform=ax.transAxes, ha='right'
        )
        
        plt.tight_layout()
        
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        plot_path = os.path.join(plots_folder, f"riskoptima_efficient_frontier_monter_carlo_{timestamp}.png")
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def add_stats_text_box(ax, title, stats_list, x=1.19, y=0.34, color='green', fontsize=10):
        """
        Adds a styled text box with statistics to the plot.

        :param ax: The matplotlib Axes object where the text box will be added.
        :param title: The title of the text box (e.g., "ML & MV Optimized Portfolio").
        :param stats_list: A list of strings with the statistics to display.
        :param x: The x-coordinate of the text box in Axes coordinates (default: 1.19).
        :param y: The y-coordinate of the text box in Axes coordinates (default: 0.34).
        :param color: The edge color of the text box (default: 'green').
        :param fontsize: The font size for the text (default: 10).
        """
        stats_text = title + ":\n" + "\n".join(stats_list)
        ax.text(
            x, y,
            stats_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor='white'),
            ha='left'
        )

    @staticmethod
    def add_ratio_explanation(ax, x=0.02, y=0.02, fontsize=10):
        """
        Adds an explanation for Sharpe Ratio, Sortino Ratio, and Info Ratio to the chart.

        :param ax: The matplotlib Axes object where the explanation will be added.
        :param x: The x-coordinate of the text box in Axes coordinates (default: 0.02).
        :param y: The y-coordinate of the text box in Axes coordinates (default: 0.02).
        :param fontsize: Font size for the text (default: 10).
        """
        explanation_text = (
            "Ratio Explanations:\n"
            "1. Sharpe Ratio: Measures risk-adjusted return using total volatility. Higher is better.\n"
            "2. Sortino Ratio: Focuses on downside risk-adjusted returns. Higher is better.\n"
            "3. Info Ratio: Measures portfolio performance vs. a benchmark. Higher is better."
        )
        ax.text(
            x, y,
            explanation_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'),
            ha='left'
        )

    @staticmethod
    def setup_chart_aesthetics(start_date="2023-12-01", end_date="2025-01-01"):
        """
        Helper method to set up chart aesthetics.
        """
        sns.set_palette("bright")
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.subplots_adjust(right=0.95)
        ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        colors = sns.color_palette()

        major_locator = AutoDateLocator(minticks=10, maxticks=15)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(AutoDateLocator(minticks=20, maxticks=30))
        ax.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d"))

        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.4, color='lightgray', alpha=0.5)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2f}%'.format(y)))

        # Add an external border around the chart
        rect = patches.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,  # Normalized coordinates
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        return ax, plt, colors

    @staticmethod
    def generate_predictions_tickers(tickers, start_date, end_date, model_type):
        """
        Generate stock predictions for multiple tickers
        """
        predicted_return = {}
        model_confidence = {}
        for ticker in tickers:
            predicted_return[ticker], model_confidence[ticker] = RiskOptima.generate_stock_predictions(
                ticker, start_date, end_date, model_type=model_type
            )
        return predicted_return, model_confidence

    @staticmethod
    def calculate_performance_metrics(portfolio_returns, market_returns, risk_free_rate, 
                                      final_returns, is_market_return=False):
        """
        Helper function to produce lines of text with performance metrics
        """
        if is_market_return:
            return [
                f"Sharpe Ratio: {RiskOptima.sharpe_ratio(portfolio_returns, risk_free_rate).iloc[0]:.2f}",
                f"Sortino Ratio: {RiskOptima.sortino_ratio(portfolio_returns, risk_free_rate).iloc[0]:.2f}",
                f"Info Ratio: {RiskOptima.information_ratio(portfolio_returns, market_returns):.2f}",
                f"Return: {final_returns:.2f}%"
            ]
        return [
            f"Sharpe Ratio: {RiskOptima.sharpe_ratio(portfolio_returns, risk_free_rate):.2f}",
            f"Sortino Ratio: {RiskOptima.sortino_ratio(portfolio_returns, risk_free_rate):.2f}",
            f"Info Ratio: {RiskOptima.information_ratio(portfolio_returns, market_returns):.2f}",
            f"Return: {final_returns:.2f}%"
        ]

    @staticmethod
    def plot_performance_metrics(model_type,
                                 portfolio_returns_unoptimized,
                                 portfolio_returns_mv,
                                 portfolio_returns_ml_mv,
                                 market_returns,
                                 risk_free_rate,
                                 final_returns_unoptimized,
                                 final_returns_mv,
                                 final_returns_ml_mv,
                                 final_returns_market,
                                 ax,
                                 column_colors):
        """
        Plots a table of performance metrics for different strategies.
        """
        titles = [
            "Unoptimized\nPortfolio",
            "Mean-Variance\nOptimized Portfolio",
            f"{model_type} & Mean-Variance\nOptimized Portfolio",
            "Benchmark\n(S&P 500)"
        ]
        stats_lists = [
            RiskOptima.calculate_performance_metrics(portfolio_returns_unoptimized, market_returns, risk_free_rate, final_returns_unoptimized),  
            RiskOptima.calculate_performance_metrics(portfolio_returns_mv, market_returns, risk_free_rate, final_returns_mv),
            RiskOptima.calculate_performance_metrics(portfolio_returns_ml_mv, market_returns, risk_free_rate, final_returns_ml_mv),
            RiskOptima.calculate_performance_metrics(market_returns, market_returns, risk_free_rate, final_returns_market, True)
        ]
        stats_df = RiskOptima.consolidate_stats_to_dataframe(titles, stats_lists)
        RiskOptima.add_table_to_plot(ax, stats_df, None, column_colors, x=1.02, y=0.30, fontsize=9)


    @staticmethod
    def run_portfolio_analysis(
        investment_allocation,
        analysis_start_date,
        analysis_end_date,
        benchmark_index='SPY',
        risk_free_rate=4.611 / 100,
        number_of_portfolio_weights=10_000,
        trading_days_per_year=260,
        number_of_monte_carlo_runs=1_000
    ):
        """
        perform portfolio analysis with Monte Carlo simulations and MPT.
        
        :param investment_allocation: dict of ticker -> investment (e.g. {'AAPL':1500,'JNJ':1200,...})
        :param analysis_start_date: Start date for historical data (YYYY-MM-DD)
        :param analysis_end_date: End date for historical data (YYYY-MM-DD)
        :param benchmark_index: e.g. 'SPY'
        :param risk_free_rate: e.g. 0.05
        :param number_of_portfolio_weights: Monte Carlo sample size for random weights
        :param trading_days_per_year: Typically 252 or 260
        :param number_of_monte_carlo_runs: e.g. 1000
        :return: Dictionary with all relevant results and data
        """

        stock_tickers, initial_weights = RiskOptima.calculate_portfolio_allocation(investment_allocation)

        stock_data = RiskOptima.download_data_yfinance(stock_tickers, analysis_start_date, analysis_end_date)
        benchmark_data = RiskOptima.download_data_yfinance([benchmark_index], analysis_start_date, analysis_end_date)

        if isinstance(benchmark_data, pd.DataFrame) and benchmark_index in benchmark_data.columns:
            benchmark_data = benchmark_data[benchmark_index]

        # Calculate daily returns
        stock_daily_returns = stock_data.pct_change(fill_method=None).dropna()
        benchmark_daily_returns = benchmark_data.pct_change(fill_method=None).dropna()

        # Generate covariance matrix for the stock returns
        covariance_matrix = stock_daily_returns.cov()

        # Initialize arrays for simulation results and recorded weights
        simulation_results = np.zeros((4, number_of_portfolio_weights))
        recorded_weights = np.zeros((len(stock_tickers), number_of_portfolio_weights))

        # Monte Carlo Simulation
        for i in range(number_of_portfolio_weights):
            # Generate random weights and normalize
            random_weights = np.random.random(len(stock_tickers))
            normalized_weights = random_weights / np.sum(random_weights)
            recorded_weights[:, i] = normalized_weights

            annualized_return = np.sum(
                normalized_weights * stock_daily_returns.mean()
            ) * trading_days_per_year
            annualized_stddev = np.sqrt(
                np.dot(normalized_weights.T, np.dot(covariance_matrix, normalized_weights))
            ) * np.sqrt(trading_days_per_year)
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_stddev

            simulation_results[:, i] = [annualized_return, annualized_stddev, sharpe_ratio, i]

        columns = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Simulation Index']
        simulated_portfolios = pd.DataFrame(simulation_results.T, columns=columns)

        # Sort by volatility, find portfolio with maximum Sharpe ratio and median volatility
        sorted_by_volatility = simulated_portfolios.sort_values(by='Annualized Volatility').reset_index()
        optimal_sharpe_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
        median_volatility_idx = sorted_by_volatility.iloc[len(sorted_by_volatility) // 2]['Simulation Index']

        optimal_weights = recorded_weights[:, optimal_sharpe_idx]
        #optimal_weights_percent = optimal_weights * 100
        #optimal_weights_percent_str = ', '.join([f"{weight:.2f}%" for weight in optimal_weights_percent])
        median_volatility_weights = recorded_weights[:, int(median_volatility_idx)]

        # Prepare for distribution simulation
        daily_mean_returns = stock_daily_returns.mean()
        #daily_volatility = stock_daily_returns.std()
        benchmark_mean_return = benchmark_daily_returns.mean()
        benchmark_volatility = benchmark_daily_returns.std()

        portfolio_weights = {
            'Optimized Portfolio': optimal_weights,
            'Current Portfolio': initial_weights,
            'Median Portfolio': median_volatility_weights
        }
        portfolio_results = {name: [] for name in portfolio_weights.keys()}
        market_final_values = []

        def run_simulation(weights, length, covariance_matrix):
            """Runs a Monte Carlo simulation for a given set of weights and time period, considering asset correlation."""
            fund_value = [10000]
            chol_matrix = np.linalg.cholesky(covariance_matrix)
            for _ in range(length):
                correlated_random_returns = np.dot(chol_matrix, np.random.normal(size=(len(stock_tickers),)))
                individual_asset_returns = daily_mean_returns + correlated_random_returns
                portfolio_return = np.dot(weights, individual_asset_returns)
                fund_value.append(fund_value[-1] * (1 + portfolio_return))
            return fund_value

        portfolio_metrics = {}
        for portfolio_name, weights in portfolio_weights.items():
            final_values = []
            returns_list = []
            for _ in range(number_of_monte_carlo_runs):
                simulated_fund_values = run_simulation(weights, trading_days_per_year, covariance_matrix)
                final_value = simulated_fund_values[-1]
                final_values.append(final_value)
                simulation_return = (final_value / 10000) - 1
                returns_list.append(simulation_return)
            portfolio_results[portfolio_name] = final_values
            expected_return = np.mean(returns_list)
            volatility = np.std(returns_list)
            portfolio_metrics[portfolio_name] = (expected_return, volatility)

        for _ in range(number_of_monte_carlo_runs):
            market_fund_value = [10000]
            for _ in range(trading_days_per_year):
                mr = np.random.normal(benchmark_mean_return, benchmark_volatility)
                market_fund_value.append(market_fund_value[-1] * (1 + mr))
            market_final_values.append(market_fund_value[-1])

        market_final_values_percent = [(value / 10000 - 1) * 100 for value in market_final_values]
        market_expected_return = np.mean(market_final_values) / 10000 - 1
        market_vol = np.std(market_final_values) / 10000
        market_sharpe_ratio = (market_expected_return - risk_free_rate) / market_vol

        return {
            "stock_tickers": stock_tickers,
            "initial_weights": initial_weights,
            "stock_data": stock_data,
            "benchmark_data": benchmark_data,
            "stock_daily_returns": stock_daily_returns,
            "benchmark_daily_returns": benchmark_daily_returns,
            "covariance_matrix": covariance_matrix,
            "simulated_portfolios": simulated_portfolios,
            "recorded_weights": recorded_weights,
            "optimal_weights": optimal_weights,
            "optimal_sharpe_idx": optimal_sharpe_idx,
            "median_volatility_weights": median_volatility_weights,
            "portfolio_results": portfolio_results,
            "portfolio_metrics": portfolio_metrics,
            "market_final_values": market_final_values,
            "market_final_values_percent": market_final_values_percent,
            "market_expected_return": market_expected_return,
            "market_volatility": market_vol,
            "market_sharpe_ratio": market_sharpe_ratio
        }

    @staticmethod
    def plot_weights_table(initial_weights, optimal_weights, labels, ax):
        """Plot Table 1: Weights Comparison Table."""
        df_weights = pd.DataFrame(columns=["Fund", "Current Weights", "Optimal Weights"])
        if initial_weights is not None and optimal_weights is not None:
            curr_w_str = [f"{w*100:.2f}%" for w in initial_weights]
            opt_w_str = [f"{w*100:.2f}%" for w in optimal_weights]
            for label, cw, ow in zip(labels, curr_w_str, opt_w_str):
                df_weights.loc[len(df_weights)] = [label, cw, ow]
        df_weights.set_index("Fund", inplace=True)
        RiskOptima.add_table_to_plot(
            ax, df_weights, x=1.02, y=0.52, column_width=0.40, fontsize=9
        )

    @staticmethod
    def plot_performance_table(portfolio_metrics, benchmark_index, market_data, risk_free_rate, ax):
        """Plot Table 2: Performance Metrics Table."""
        market_expected_return, market_volatility, market_sharpe_ratio = market_data
        perf_data = []
        for portfolio_name, (mean_val, vol_val) in portfolio_metrics.items():
            sharpe_val = (mean_val - risk_free_rate) / vol_val if vol_val != 0 else 0
            perf_data.append([
                portfolio_name,
                f"{mean_val*100:.2f}%",
                f"{vol_val*100:.2f}%",
                f"{sharpe_val:.2f}"
            ])
        perf_data.append([
            benchmark_index,
            f"{market_expected_return*100:.2f}%",
            f"{market_volatility*100:.2f}%",
            f"{market_sharpe_ratio:.2f}"
        ])
        df_perf = pd.DataFrame(perf_data, columns=["Portfolio", "Mean%", "Vol%", "Sharpe"])
        df_perf.set_index("Portfolio", inplace=True)
        RiskOptima.add_table_to_plot(
            ax, df_perf, x=1.02, y=0.28, column_width=0.40, fontsize=9
        )

    @staticmethod
    def plot_probability_table(portfolio_results, market_final_values, ax):
        """Plot Table 3: Probability Comparison Table."""
        optimized_vals = portfolio_results.get("Optimized Portfolio", [])
        current_vals = portfolio_results.get("Current Portfolio", [])
        prob_text_data = []
    
        if optimized_vals and current_vals:
            p_beat_current = sum(np.array(optimized_vals) > np.array(current_vals)) / len(optimized_vals)
            prob_text_data.append(["Prob(Optimized > Current)", f"{p_beat_current:.2%}"])
    
        if optimized_vals and market_final_values:
            p_beat_market = sum(np.array(optimized_vals) > np.array(market_final_values)) / len(optimized_vals)
            prob_text_data.append(["Prob(Optimized > Market)", f"{p_beat_market:.2%}"])
    
        df_prob = pd.DataFrame(prob_text_data, columns=["Description", "Value"])
        df_prob.set_index("Description", inplace=True)
        RiskOptima.add_table_to_plot(
            ax, df_prob, x=1.02, y=0.12, column_width=0.40, fontsize=9
        )
       
    @staticmethod
    def create_portfolio_area_chart(
        asset_table,
        end_date=None, 
        lookback_days=5, 
        title="Portfolio Area Chart"
    ):
        """
        Create a market-area chart with a gradient colour scheme that depends on 
        the percentage change over a specified lookback period,
        showing each asset's return and allocation percentage.
        """
    
        assets = asset_table["Asset"].tolist()
        weights = asset_table["Weight"].to_numpy()
        labels = asset_table["Label"].tolist()
    
        if end_date:
            end_dt = pd.to_datetime(end_date)
            start_dt = end_dt - pd.Timedelta(days=(lookback_days + 10))
            data = yf.download(
                assets,
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                interval="1d",
                progress=False
            )
        else:
            data = yf.download(assets, period="1mo", interval="1d", progress=False)
    
        close_prices = data["Close"]
        if len(close_prices) < lookback_days:
            raise ValueError(f"Not enough data to compute {lookback_days}-day returns.")
    
        recent = close_prices.iloc[-1]
        previous = close_prices.iloc[-lookback_days]
        pct_change = ((recent - previous) / previous) * 100
        pct_change = pct_change.fillna(0)
    
        t_minus_x_data = pd.DataFrame({
            "Asset": assets,
            f"Close(T-{lookback_days})": previous.values,
            "Close(T)": recent.values,
            f"{lookback_days}d % Change": pct_change.values
        })
        print(t_minus_x_data)
    
        latest_date = close_prices.index[-1].strftime('%Y-%m-%d')
        full_title = f"[RiskOptima] {title}: {lookback_days}-Day Returns as of {latest_date}"
    
        assert len(weights) == len(assets), "Weights array length must match the number of assets"
    
        min_val = pct_change.min()
        max_val = pct_change.max()
    
        if min_val == max_val:
            min_val = max_val - 0.0001
    
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "greyredgreen", 
            [ (0.7, 0.7, 0.7),  # grey at "centre"
              (1.0, 0.0, 0.0)   # red 
            ],
            N=256
        )
        cmap_green = mpl.colors.LinearSegmentedColormap.from_list(
            "greygreen", 
            [ (0.7, 0.7, 0.7),  # grey
              (0.0, 1.0, 0.0)   # green
            ],
            N=256
        )
    
        def blended_colour(value):
            if max_val <= 0:
                ratio = (value - min_val) / (0 - min_val)
                ratio = np.clip(ratio, 0, 1)
                return cmap(ratio)
            elif min_val >= 0:
                ratio = (value - 0) / (max_val - 0)
                ratio = np.clip(ratio, 0, 1)
                return cmap_green(ratio)
            else:
                if value < 0:
                    ratio = (value - min_val) / (0 - min_val)
                    ratio = np.clip(ratio, 0, 1)
                    return cmap(ratio)
                else:
                    ratio = (value - 0) / (max_val - 0)
                    ratio = np.clip(ratio, 0, 1)
                    return cmap_green(ratio)
    
        colours = [blended_colour(v) for v in pct_change]
    
        labels = [
            f"{name}\n"
            f"{'+' if ret > 0 else ''}{ret:.2f}%\n"
            f"Allocation: {w * 100:.1f}%"
            for name, ret, w in zip(labels, pct_change, weights)
        ]
    
        sizes = weights * 100
    
        fig, ax = plt.subplots(figsize=(18, 12))
        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colours,
            alpha=0.8,
            ax=ax,
            edgecolor="#4f4f4f",  # dark grey border
            linewidth=2
        )
        ax.set_title(full_title, fontsize=18)
        ax.axis('off')
    
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        combined_cmap = mpl.cm.RdYlGn
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = mpl.cm.ScalarMappable(cmap=combined_cmap, norm=norm)
        sm.set_array([])  # required for matplotlib < 3.2
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f"{lookback_days}-Day Return (%)", fontsize=12)
        
        
        plt.text(
            0.995, -0.05, f"Created by RiskOptima v{RiskOptima.VERSION}",
            fontsize=12, color='gray', alpha=0.7, transform=ax.transAxes, ha='right'
        )
        
        plots_folder = "plots"
        
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_folder, f"riskoptima_portfolio_area_chart_{timestamp}.png")
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    @staticmethod
    def run_portfolio_optimization_mv_ml(
        asset_table, 
        training_start_date, 
        training_end_date,
        model_type='Linear Regression',
        risk_free_rate=0.05,
        num_portfolios=100000,
        market_benchmark=['SPY'],
        max_volatility=0.15,
        min_weight=0.03,
        max_weight=0.2
    ):
        """
        Run portfolio optimization using machine learning and mean-variance optimization.
        
        Parameters:
            asset_table (pd.DataFrame): A DataFrame with columns:
                - 'Asset': Ticker symbol.
                - 'Weight': Original portfolio weight (as a fraction).
                - 'Label': Descriptive name.
                - 'MarketCap': Market capitalization for the asset.
            model_type (str): The model to use (e.g. 'Linear Regression').
            training_start_date (str): Start date for training data in 'YYYY-MM-DD' format.
            training_end_date (str): End date for training data in 'YYYY-MM-DD' format.
            risk_free_rate (float): Risk-free rate used in performance calculations.
            num_portfolios (int): (Optional) Number of portfolios to simulate (if needed).
            market_benchmark (list): List of market benchmark tickers (e.g. ['SPY']).
            max_volatility (float): Maximum allowed portfolio volatility.
            min_weight (float): Minimum allocation weight per asset.
            max_weight (float): Maximum allocation weight per asset.
        
        The function performs the following steps:
          - Builds a portfolio dictionary from the asset table (multiplying weights by 100,000).
          - Extracts market cap information.
          - Performs mean-variance optimization (both standard and adjusted via a machine learning model).
          - Fetches historical price data and computes daily and cumulative returns.
          - Sets up an aesthetic chart with tables comparing original and optimized weights.
          - Plots cumulative returns for the original, optimized, benchmark, and ML-adjusted portfolios.
          - Annotates the plot with performance metrics.
          
        Finally, the chart is saved to disk and displayed.
        """
        # Build a portfolio dictionary (dollar amounts)
        portfolio = {
            row['Asset']: row['Weight'] * 100000 
            for _, row in asset_table.iterrows()
        }
        # Build market caps dictionary from the asset table
        market_caps = {
            row['Asset']: row['MarketCap'] 
            for _, row in asset_table.iterrows()
        }
        # Get labels for display (as a numpy array)
        my_current_labels = asset_table['Label'].values
    
        # Define default table column descriptions and colors for the chart table
        column_descriptions = [
            "Security", 
            "Original\nPortfolio Weights", 
            "Mean-Variance\nOptimization", 
            f"{model_type} & Mean-Variance\nOptimization"
        ]
        column_colors = [
            "white",          # For the index column
            "#e69a9a",        # Original, unoptimized portfolio
            "#e6b89a",        # Mean-Variance optimization
            "#9ac7e6",        # Market benchmark (S&P 500)
            "#9ae69b",        # ML & Mean-Variance optimization
        ]
        
        # Calculate tickers and original weights from the portfolio dictionary.
        tickers, weights = RiskOptima.calculate_portfolio_allocation(portfolio)
        
        # --- Mean-Variance Optimization ---
        optimized_weights_mv = RiskOptima.perform_mean_variance_optimization(
            tickers, training_start_date, training_end_date, 
            max_volatility, min_allocation=min_weight, max_allocation=max_weight
        )
        
        # --- Machine Learning Prediction for Adjusted Returns ---
        investor_views, view_confidences = RiskOptima.generate_predictions_tickers(
            tickers, training_start_date, training_end_date, model_type
        )
        
        index_data = RiskOptima.fetch_historical_stock_prices(
            market_benchmark, training_start_date, training_end_date
        )
        index_return = (index_data['Close'].iloc[-1] / index_data['Close'].iloc[0]) - 1
        
        # Compute market returns for each asset based on market caps and index return
        computed_market_returns = RiskOptima.compute_market_returns(market_caps, index_return)
        
        historical_data = RiskOptima.fetch_historical_stock_prices(
            tickers, training_start_date, training_end_date
        )
        predicted_returns = RiskOptima.black_litterman_adjust_returns(
            computed_market_returns, investor_views, view_confidences, historical_data
        )
        predicted_returns = dict(zip(tickers, predicted_returns))
        adjusted_returns_vector = np.array([predicted_returns[ticker] for ticker in tickers])
        
        optimized_weights_ml_mv = RiskOptima.perform_mean_variance_optimization(
            tickers, training_start_date, training_end_date, 
            max_volatility, adjusted_returns_vector, min_weight, max_weight
        )
        
        # --- Backtesting ---
        backtesting_start_date = training_end_date
        backtesting_end_date = RiskOptima.get_previous_working_day()
        
        historical_data_backtest = RiskOptima.fetch_historical_stock_prices(
            tickers, backtesting_start_date, backtesting_end_date
        )
        # Forward-fill missing data and compute daily returns
        historical_data_filled = historical_data_backtest['Close'].ffill()
        daily_returns_backtest = historical_data_filled.pct_change()
        
        portfolio_returns_ml_mv = daily_returns_backtest.dot(optimized_weights_ml_mv)
        cumulative_returns_ml_mv = (1 + portfolio_returns_ml_mv).cumprod()
        
        portfolio_returns_mv = daily_returns_backtest.dot(optimized_weights_mv)
        cumulative_returns_mv = (1 + portfolio_returns_mv).cumprod()
        
        market_data = RiskOptima.fetch_historical_stock_prices(
            market_benchmark, backtesting_start_date, backtesting_end_date
        )['Close']
        market_data_filled = market_data.ffill()
        market_returns_series = market_data_filled.pct_change()
        cumulative_market_returns = (1 + market_returns_series).cumprod()
        
        portfolio_returns_unoptimized = daily_returns_backtest.dot(weights)
        cumulative_returns_unoptimized = (1 + portfolio_returns_unoptimized).cumprod()
        
        # Format weights as percentages for comparison
        weights_pct = [f'{w * 100:.2f}%' for w in weights]
        optimized_weights_pct = [f'{w * 100:.2f}%' for w in optimized_weights_mv]
        optimized_weights_ml_mv_pct = [f'{w * 100:.2f}%' for w in optimized_weights_ml_mv]
        
        portfolio_comparison = pd.DataFrame({
            'Original': weights_pct,
            'MV Optimization': optimized_weights_pct,
            f'{model_type} & MV Optimization': optimized_weights_ml_mv_pct
        }, index=tickers)
        portfolio_comparison.index = my_current_labels
        
        # --- Chart Setup and Plotting ---
        ax, plt_obj, _ = RiskOptima.setup_chart_aesthetics(backtesting_start_date, backtesting_end_date)
        RiskOptima.add_table_to_plot(ax, portfolio_comparison, column_descriptions, x=1.02, y=0.52)
        
        # Convert cumulative returns to percentage gain
        cumulative_returns_ml_mv_percent = (cumulative_returns_ml_mv - 1) * 100
        cumulative_returns_mv_percent = (cumulative_returns_mv - 1) * 100
        cumulative_returns_unoptimized_percent = (cumulative_returns_unoptimized - 1) * 100
        cumulative_market_returns_percent = (cumulative_market_returns - 1) * 100
        
        final_returns_ml_mv = cumulative_returns_ml_mv_percent.iloc[-1]
        final_returns_mv = cumulative_returns_mv_percent.iloc[-1]
        final_returns_unoptimized = cumulative_returns_unoptimized_percent.iloc[-1]
        final_returns_market = cumulative_market_returns_percent.iloc[-1]
        
        if isinstance(final_returns_market, pd.Series):
            final_returns_market = final_returns_market.iloc[0]
        
        # Plot curves
        ax.plot(cumulative_returns_unoptimized_percent, label='Original Unoptimized Portfolio', color=column_colors[1])
        ax.plot(cumulative_returns_mv_percent, label='Portfolio Optimized with Mean-Variance', color=column_colors[2])
        ax.plot(cumulative_market_returns_percent, label='Market Index Benchmark (S&P 500)', color=column_colors[3])
        ax.plot(cumulative_returns_ml_mv_percent, label=f'Portfolio Optimized with {model_type} and Mean-Variance', color=column_colors[4])
        
        RiskOptima.plot_performance_metrics(
            model_type, portfolio_returns_unoptimized, portfolio_returns_mv, portfolio_returns_ml_mv,
            market_returns_series, risk_free_rate, final_returns_unoptimized, final_returns_mv,
            final_returns_ml_mv, final_returns_market, ax, column_colors
        )
        
        RiskOptima.add_ratio_explanation(ax, x=1.02, y=0.01, fontsize=9)
        
        plot_title = ("[RiskOptima] Portfolio Optimization and Benchmarking: Comparing Machine Learning and Statistical "
                      "Models for Risk-Adjusted Performance")
        plt_obj.title(plot_title, fontsize=14, pad=20)
        plt_obj.xlabel('Date')
        plt_obj.ylabel('Percentage Gain (%)')
        plt_obj.legend(loc='lower center')
        plt_obj.grid(True)
        
        plt_obj.text(
            0.995, -0.15, f"Created by RiskOptima v{RiskOptima.VERSION}",
            fontsize=12, color='gray', alpha=0.7, transform=ax.transAxes, ha='right'
        )
        
        plots_folder = "plots"
        
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_folder, f"riskoptima_machine_learning_optimization_{timestamp}.png")
        
        plt_obj.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt_obj.show()        

    @staticmethod        
    def run_portfolio_probability_analysis(
        asset_table,
        analysis_start_date,
        analysis_end_date,
        benchmark_index,
        risk_free_rate,
        number_of_portfolio_weights,
        trading_days_per_year,
        number_of_monte_carlo_runs
    ):
        """
        Run portfolio probability analysis from an asset table. The asset_table is expected to have:
          - 'Asset': ticker symbol.
          - 'Weight': original portfolio weight (as a fraction).
          - 'Label': descriptive name.
          - 'MarketCap': market capitalisation.
          - 'Portfolio': the dollar allocation (e.g. Weight * capital).
          
        Additional parameters define the analysis dates, benchmark index, risk‐free rate, simulation parameters, etc.
        
        The function calls an internal RiskOptima.run_portfolio_analysis method (which uses the provided
        investment allocation) and then produces probability distribution plots and performance tables.
        """
        
        def density_as_percent(y, _):
            return f"{y*100:.2f}%"
        
        # Create the investment allocation dictionary from the asset table’s "Portfolio" column.
        investment_allocation = {row['Asset']: row['Portfolio'] for _, row in asset_table.iterrows()}
        
        # Extract the labels for later use in tables.
        my_current_labels = asset_table['Label'].values
    
        # Run the portfolio analysis.
        results = RiskOptima.run_portfolio_analysis(
            investment_allocation       = investment_allocation,
            analysis_start_date         = analysis_start_date,
            analysis_end_date           = analysis_end_date,
            benchmark_index             = benchmark_index,
            risk_free_rate              = risk_free_rate,
            number_of_portfolio_weights = number_of_portfolio_weights,
            trading_days_per_year       = trading_days_per_year,
            number_of_monte_carlo_runs  = number_of_monte_carlo_runs
        )
    
        # Extract results
        portfolio_results      = results["portfolio_results"]
        portfolio_metrics      = results["portfolio_metrics"]
        market_final_values    = results["market_final_values"]
        market_expected_return = results["market_expected_return"]
        market_volatility      = results["market_volatility"]
        market_sharpe_ratio    = results["market_sharpe_ratio"]
    
        initial_weights = results.get("initial_weights", None)
        optimal_weights = results.get("optimal_weights", None)
    
        # Convert market final values to percentages for plotting.
        market_final_values_percent = [(val/10000.0 - 1)*100 for val in market_final_values]
    
        # Set up the plot.
        fig, ax = plt.subplots(figsize=(23, 10))
        fig.subplots_adjust(right=0.80)
        ax = plt.gca()
        sns.set_style("whitegrid")
        ax.set_facecolor('white')
        plt.gcf().set_facecolor('white')
        
        # Configure axis locators and formatters.
        ax.xaxis.set_major_locator(MaxNLocator(integer=False, prune=None, nbins=20))
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune=None, nbins=15))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}%'))
        ax.yaxis.set_major_formatter(FuncFormatter(density_as_percent))
        
        ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.4, color='lightgray', alpha=0.5)
    
        # Plot distributions for each portfolio.
        palette = sns.color_palette("pastel", len(portfolio_results) + 1)
        i = 0
        for portfolio_name, final_values in portfolio_results.items():
            color = palette[i]
            final_values_percent = [(value/10000.0 - 1)*100 for value in final_values]
            sns.kdeplot(final_values_percent, label=portfolio_name, color=color, ax=ax)
            # If mean/vol metrics are available, add a vertical line.
            if portfolio_name in portfolio_metrics:
                mean_val, _ = portfolio_metrics[portfolio_name]
                ax.axvline(x=mean_val*100, color=color, linestyle='--', linewidth=1)
            i += 1
    
        # Plot the market distribution.
        market_color = palette[-1]
        sns.kdeplot(market_final_values_percent, label=benchmark_index, color=market_color, ax=ax)
        ax.axvline(x=market_expected_return*100, color=market_color, linestyle='--', linewidth=1)
    
        plt.xlabel('Final Fund % Returns')
        plt.ylabel('Density')
        plt.title(f'[RiskOptima] Probability Distributions of Final Fund Returns {analysis_start_date} to {analysis_end_date}', fontsize=14)
        plt.legend(loc='best')
        
        plt.text(
            0.995, -0.10, f"Created by RiskOptima v{RiskOptima.VERSION}",
            fontsize=12, color='gray', alpha=0.7, transform=ax.transAxes, ha='right'
        )
    
        # Plot additional tables using helper functions in RiskOptima.
        RiskOptima.plot_weights_table(initial_weights, optimal_weights, my_current_labels, ax)
        RiskOptima.plot_performance_table(
            portfolio_metrics,
            benchmark_index,
            (market_expected_return, market_volatility, market_sharpe_ratio),
            risk_free_rate,
            ax
        )
        RiskOptima.plot_probability_table(portfolio_results, market_final_values, ax)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plots_folder = "plots"
        
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        plot_path = os.path.join(plots_folder, f"riskoptima_probability_distributions_of_final_fund_returns{timestamp}.png")
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()     
        
    @staticmethod
    def black_scholes(S, X, T, r, sigma):
        """
        Computes the Black-Scholes option price for a European call option.
        
        :param float S: Current price of the underlying asset.
        :param float X: Strike price of the option.
        :param float T: Time to expiration in years.
        :param float r: Risk-free interest rate.
        :param float sigma: Volatility of the underlying asset.
        :return: The calculated call option price as a float.
        """
        d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * si.norm.cdf(d1) - X * np.exp(-r * T) * si.norm.cdf(d2)
        return call_price
    
    @staticmethod
    def heston(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0, num_simulations=10000, num_steps=100):
        """
        Computes the option price using the Heston model via Monte Carlo simulation.
        
        :param float S_0: Initial price of the underlying asset.
        :param float X: Strike price of the option.
        :param float T: Time to expiration in years.
        :param float r: Risk-free interest rate.
        :param float kappa: Mean reversion rate of volatility.
        :param float theta: Long-term mean of volatility.
        :param float sigma_v: Volatility of volatility.
        :param float rho: Correlation between asset return and volatility.
        :param float v_0: Initial variance.
        :param int num_simulations: Number of simulations (default: 10000).
        :param int num_steps: Number of steps in the simulation (default: 100).
        :return: The estimated option price using the Heston model as a float.
        """
        dt = T / num_steps
        option_payoffs = []
    
        for _ in range(num_simulations):
            S_t = S_0
            v_t = v_0
            for _ in range(num_steps):
                z1, z2 = np.random.normal(size=2)
                z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
                S_t += r * S_t * dt + np.sqrt(v_t) * S_t * z1 * np.sqrt(dt)
                v_t += kappa * (theta - v_t) * dt + sigma_v * np.sqrt(v_t) * z2 * np.sqrt(dt)
                v_t = max(v_t, 0)
            option_payoff = max(S_t - X, 0)
            option_payoffs.append(option_payoff)
    
        average_payoff = np.mean(option_payoffs)
        option_price = np.exp(-r * T) * average_payoff
    
        return option_price
    
    @staticmethod
    def merton_jump_diffusion(S_0, X, T, r, sigma, lambda_jump, m_jump, delta_jump, num_simulations=10000, num_steps=100):
        """
        Computes the option price using the Merton Jump Diffusion model via Monte Carlo simulation.
        
        :param float S_0: Initial price of the underlying asset.
        :param float X: Strike price of the option.
        :param float T: Time to expiration in years.
        :param float r: Risk-free interest rate.
        :param float sigma: Volatility of the underlying asset.
        :param float lambda_jump: Intensity of the jumps.
        :param float m_jump: Mean of the jump size.
        :param float delta_jump: Volatility of the jump size.
        :param int num_simulations: Number of simulations (default: 10000).
        :param int num_steps: Number of steps in the simulation (default: 100).
        :return: The estimated option price using the Merton Jump Diffusion model as a float.
        """
        dt = T / num_steps
        S_t = np.full(num_simulations, S_0, dtype=np.float64)
    
        for _ in range(num_steps):
            z = np.random.normal(size=num_simulations)
            jump_sizes = np.random.normal(loc=m_jump, scale=delta_jump, size=num_simulations)
            jumps = np.random.poisson(lambda_jump * dt, size=num_simulations)
            S_t *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            S_t *= np.exp(jumps * jump_sizes)
    
        option_payoffs = np.maximum(S_t - X, 0)
        average_payoff = np.mean(option_payoffs)
        option_price = np.exp(-r * T) * average_payoff
    
        return option_price
    
    @staticmethod
    def create_heatmap(S_0, X, T, lambda_jump, m_jump, delta_jump,
                       volatility_range=(0.1, 0.5), interest_rate_range=(0.01, 0.1),
                       volatility_steps=12, interest_rate_steps=12, sigma=0.25):
        """
        Generates a heatmap of option prices using the Merton Jump Diffusion model.
        
        :param float S_0: Initial price of the underlying asset.
        :param float X: Strike price of the option.
        :param float T: Time to expiration in years.
        :param float lambda_jump: Intensity of the jumps.
        :param float m_jump: Mean of the jump size.
        :param float delta_jump: Volatility of the jump size.
        :param tuple volatility_range: Range of volatilities to iterate over (default: (0.1, 0.5)).
        :param tuple interest_rate_range: Range of interest rates to iterate over (default: (0.01, 0.1)).
        :param int volatility_steps: Number of steps in the volatility grid (default: 12).
        :param int interest_rate_steps: Number of steps in the interest rate grid (default: 12).
        """
        volatility_grid = np.linspace(volatility_range[0], volatility_range[1], volatility_steps)
        interest_rate_grid = np.linspace(interest_rate_range[0], interest_rate_range[1], interest_rate_steps)
        
        option_prices_matrix = np.zeros((len(interest_rate_grid), len(volatility_grid)))
        
        for i, r in enumerate(interest_rate_grid):
            for j, sigma_v in enumerate(volatility_grid):
                option_prices_matrix[i, j] = RiskOptima.merton_jump_diffusion(S_0, X, T, r, sigma, lambda_jump, m_jump, delta_jump)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(option_prices_matrix, cmap='viridis', extent=[volatility_grid[0], volatility_grid[-1], interest_rate_grid[0], interest_rate_grid[-1]], aspect='auto', origin='lower')
        plt.colorbar(label='Option Price')
        plt.title('Option Prices Heatmap')
        plt.xlabel('Volatility (%)')
        plt.ylabel('Interest Rate (%)')
        plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        plt.show()
        
    @staticmethod
    def modified_duration(macaulay_duration, discount_rate, coupons_per_year=2):
        """
        Computes the modified duration from Macaulay duration.
        
        :param macaulay_duration: The Macaulay duration of the bond.
        :param discount_rate: The bond's yield to maturity (YTM) as a decimal.
        :param coupons_per_year: Number of coupon payments per year (default is 2 for semi-annual).
        :return: Modified duration value.
        """
        return macaulay_duration / (1 + discount_rate / coupons_per_year)

    @staticmethod
    def dollar_duration(modified_duration, bond_price, face_value=100):
        """
        Computes the dollar duration of the bond.
        
        :param modified_duration: The modified duration of the bond.
        :param bond_price: The current price of the bond.
        :param face_value: The face value of the bond (default 100).
        :return: Dollar duration value.
        """
        return modified_duration * bond_price / face_value

    @staticmethod
    def pvpb(dollar_duration):
        """
        Computes the Price Value of a Basis Point (PVBP), also known as DV01.
        
        :param dollar_duration: The dollar duration of the bond.
        :return: PVBP (Price Value of a Basis Point).
        """
        return dollar_duration / 100

    @staticmethod
    def convexity(cash_flows, discount_rate, bond_price):
        """
        Computes the convexity of a bond.
        
        :param cash_flows: A Pandas Series of bond cash flows indexed by time.
        :param discount_rate: The bond's yield to maturity (YTM) as a decimal.
        :param bond_price: The current price of the bond.
        :return: Convexity value.
        """
        times = cash_flows.index
        discounted_cf = cash_flows / (1 + discount_rate) ** times
        convexity = sum(discounted_cf * times * (times + 1)) / (bond_price * (1 + discount_rate) ** 2)
        return convexity