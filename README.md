# NIFTY-50-Stock-Market-Data-2000-2021-Analysis-ML-Modeling
Core Focus: This project analyzes daily OHLCV (Open, High, Low, Close, Volume) data for all Nifty 50 constituents from January 2000 to December 2021

Project Overview
The code tackles a problem statement: examine Nifty 50 index data from the Indian stock market, focusing on volatility drivers over the last decade, select promising stocks, create performance dashboards, and apply machine learning for predictions. It combines multiple CSV files (excluding NIFTY50all.csv and stockmetadata.csv) into a unified DataFrame for analysis.
​

Key Features
Data Loading and Cleaning: Uploads and unzips data, reads CSV files per stock, converts dates, removes duplicates/NaNs, forward-fills prices (Open, High, Low, Close, Volume, VWAP), and sorts by symbol/date.
​
Volatility Analysis: Computes returns, 30-day rolling volatility, daily range %, absolute returns, 50/200-day moving averages; normalizes and ranks stocks by combined volatility score (top volatile: SSLT, VEDL; least: stability picks).

​Performance Metrics: Calculates CAGR, annualized volatility, Sharpe ratio per stock; ranks top performers.
​
Visual Dashboard
Generates matplotlib/seaborn plots and Plotly interactives:

Historical close trends for selected stocks (volatile, stable, high-Sharpe).

Bar charts for normalized volatility metrics, Sharpe ratios, combined scores.

Line plots for price trends with MAs and 30-day volatility.
​
ML Pipeline
Feature Engineering: Momentum (10D), volume change, price range, volatility (30D); target is binary future 30-day return direction.
​
Model: Trains RandomForestClassifier (200 trees, max_depth=6) on time-series split (80/20); evaluates with classification report.
​
Setup and Run
Requires Google Colab (for upload/unzip); libraries: pandas, numpy, matplotlib, seaborn, sklearn, plotly. Assumes 'content/archive(1).zip' with stock CSVs. Run sequentially for data prep → analysis → viz → ML.
