import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import torch
from scipy.special import softmax
import sys
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.impute import SimpleImputer
import re
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats
def visualize_sentiment_analysis(df, sentiment, independent_var, ttest_p_values, anova_model=None, ):
    """
    Generalized function to visualize sentiment analysis results across different independent variables.

    Args:
    df (pd.DataFrame): Dataframe containing sentiment and independent variables.
    sentiment (str): Column name for sentiment label.
    independent_var (str): Column name for the independent variable (e.g., 'Gender', 'Location', 'Interest').
    ttest_p_values (tuple): Tuple containing p-values for t-test comparisons.
    anova_model (statsmodels): Fitted model object from ANOVA analysis (optional).
    """
    # Boxplot
    plot_sentiment_boxplots(df, sentiment, independent_var, save_path=None)
    
    # Barplot
    plot_mean_sentiment_bars(df, sentiment, independent_var, save_path=None)
    
    # T-test Results
    plot_ttest_results(ttest_p_values,sentiment, independent_var, save_path=None)
    
    # ANOVA residuals if available
    if anova_model:
        plot_anova_residuals(anova_model,sentiment, independent_var, save_path=None)
    
    # Tukey HSD
    plot_tukey_hsd(df, sentiment, independent_var, save_path=None)

def plot_sentiment_boxplots(df, sentiment, independent_var, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=independent_var, y=sentiment, data=df)
    plt.title(f'Boxplot of {sentiment.replace("_score", "").capitalize()} Sentiment by {independent_var}')
    plt.ylabel(f'{sentiment.replace("_score", "").capitalize()} Sentiment Score')
    plt.xlabel(independent_var)
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()

def plot_mean_sentiment_bars(df, sentiment, independent_var, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=independent_var, y=sentiment, data=df, errorbar=None)
    plt.title(f'Mean {sentiment.replace("_score", "").capitalize()} Sentiment by {independent_var}')
    plt.ylabel(f'Mean {sentiment.replace("_score", "").capitalize()} Sentiment Score')
    plt.xlabel(independent_var)
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()


def plot_ttest_results(p_values,sentiment, independent_var, save_path=None):
    ttest_results = {
        "Comparison": ["Comparison 1", "Comparison 2", "Comparison 3"],  # Update as necessary
        "p-value": p_values
    }
    ttest_df = pd.DataFrame(ttest_results)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Comparison", y="p-value", data=ttest_df)
    plt.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')
    plt.title('Pairwise T-test Results')
    plt.ylabel('p-value')
    plt.legend()
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()

def plot_anova_residuals(model, sentiment, independent_var, save_path=None):
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(8, 6))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()

def plot_tukey_hsd(df, sentiment, independent_var, save_path=None):
    tukey = pairwise_tukeyhsd(df[sentiment], df[independent_var])
    tukey.plot_simultaneous()
    plt.title(f'Tukey HSD Test for {sentiment.replace("_score", "").capitalize()} Sentiment by {independent_var}')
    plt.xlabel('Mean Difference')
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()

    
