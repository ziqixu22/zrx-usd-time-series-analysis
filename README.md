# ZRX/USD Time Series & Predictive Modeling

A team project analyzing and forecasting the ZRX/USD cryptocurrency exchange rate using regression with autocorrelated errors and ARIMA-based residual modeling.

## Project Overview

The analysis studies daily ZRX/USD market data from **January 1, 2023 to February 27, 2024**, using the daily midpoint price as the response variable and market variables such as date, bid price, ask price, and trading volume as predictors.

The project combines cross-sectional regression logic with time-series diagnostics. A key modeling challenge is that ordinary least-squares residuals are serially correlated, so the final approach augments regression with an **ARIMA(0,1,1)** error structure.

## Research Questions

1. Which market variables are associated with the ZRX/USD midpoint exchange rate?
2. Are ordinary linear-regression residuals independent over time?
3. Can ARIMA-based residual modeling remove remaining serial dependence?
4. Does a model using the ask price outperform a model using the bid price?
5. What short-term exchange-rate direction is implied by the selected model?

## Data

The project uses `ZRXUSD.csv`, with variables including:

- `date` — trading date
- `high` — daily high
- `low` — daily low
- `mid` — daily average/midpoint price
- `last` — last traded price
- `bid` — buying price at end of day
- `ask` — selling price at end of day
- `volume` — trading volume

The response variable is

$$
Y_t = \text{mid}_t.
$$

## Exploratory Time-Series Analysis

The raw midpoint-price series showed changes in level and variance over time, suggesting non-stationarity.

A first difference was therefore examined:

$$
\Delta Y_t = Y_t - Y_{t-1}.
$$

The Augmented Dickey-Fuller test on the differenced midpoint series reported

$$
\text{ADF statistic}=-9.074,\qquad p=0.01,
$$

supporting stationarity after differencing.

## Model 1: Bid-Price Regression

The first regression specification is

$$
Y_t
=\beta_0
+\beta_1\,\text{date}_t
+\beta_2\,\text{bid}_t
+\beta_3\,\text{volume}_t
+\varepsilon_t.
$$

The original fitted coefficients were approximately

$$
\hat\beta_{\text{date}}=-3.540\times10^{-7},
$$

$$
\hat\beta_{\text{bid}}\approx1.000,
$$

$$
\hat\beta_{\text{volume}}=6.276\times10^{-11}.
$$

All three predictors were reported as statistically significant at the 5% level.

### Residual Dependence

For an adequate time-series regression, residuals should approximately behave like white noise:

$$
\varepsilon_t\overset{iid}{\sim}N(0,\sigma^2).
$$

The Ljung-Box test evaluates

$$
H_0:\rho_1=\rho_2=\cdots=\rho_h=0.
$$

For the plain regression model, the reported p-value was

$$
p<2.2\times10^{-16},
$$

which strongly rejected residual independence.

## ARIMA Residual Modeling

Differencing the regression residuals produced a more stationary-looking series. The residual ACF cut off near lag 1 while the PACF tailed off, motivating an MA(1) component after one difference.

The residual process was modeled as

$$
\text{ARIMA}(0,1,1).
$$

Using the backshift operator $B$, this can be written as

$$
(1-B)e_t=(1+\theta B)w_t,
$$

where

$$
w_t\sim WN(0,\sigma_w^2).
$$

The resulting regression-with-ARIMA-errors model is conceptually

$$
Y_t=X_t^\top\beta+e_t,
$$

with $e_t$ following ARIMA$(0,1,1)$ dynamics.

For Model 1.1, the exogenous regressors are date, bid, and volume:

$$
X_t=(\text{date}_t,\text{bid}_t,\text{volume}_t).
$$

After incorporating the ARIMA error structure, the Ljung-Box p-value increased to

$$
p=0.4642,
$$

so the residuals no longer showed clear evidence of serial dependence.

## Model 2: Ask-Price Regression

The second specification replaces bid with ask:

$$
Y_t
=\beta_0
+\beta_1\,\text{date}_t
+\beta_2\,\text{ask}_t
+\beta_3\,\text{volume}_t
+\varepsilon_t.
$$

The reported coefficients were approximately

$$
\hat\beta_{\text{date}}=3.571\times10^{-7},
$$

$$
\hat\beta_{\text{ask}}=0.9996,
$$

$$
\hat\beta_{\text{volume}}=-6.214\times10^{-11}.
$$

Again, the predictors were reported as statistically significant.

The plain regression residuals exhibited strong serial dependence, with Ljung-Box

$$
p<2.2\times10^{-16}.
$$

After fitting ARIMA$(0,1,1)$ errors with date, ask, and volume as exogenous regressors, the residual Ljung-Box p-value became

$$
p=0.4721,
$$

indicating a substantial improvement in residual independence.

## Model Evaluation

The two ARIMA-regression specifications were compared using AIC, BIC, and residual diagnostics.

### Akaike Information Criterion

$$
AIC=-2\log L+2k,
$$

where $L$ is the maximized likelihood and $k$ is the number of estimated parameters.

Reported values:

$$
AIC_{1.1}=-6303.150,
$$

$$
AIC_{2.1}=-6305.717.
$$

Lower is preferred, so Model 2.1 performed slightly better by AIC.

### Bayesian Information Criterion

$$
BIC=-2\log L+k\log n.
$$

Reported values:

$$
BIC_{1.1}=-6282.996,
$$

$$
BIC_{2.1}=-6285.564.
$$

Model 2.1 also had the lower BIC.

### Residual Diagnostics

Both ARIMA-regression models showed much cleaner residual behavior than the plain OLS regressions. In particular, Ljung-Box tests failed to reject independence for both final models.

Because Model 2.1 had slightly lower AIC and BIC while also passing the residual-independence check, it was selected as the preferred specification.

## Forecasting

The preferred model combines:

$$
\text{mid}_t
=\beta_0+\beta_1\text{date}_t+\beta_2\text{ask}_t+\beta_3\text{volume}_t+e_t,
$$

with

$$
e_t\sim ARIMA(0,1,1).
$$

The project generated a short-horizon forecast using the final regression structure together with forecasted ARIMA residuals.

The original project conclusion was that the selected model implied a **declining short-term trend in the ZRX/USD midpoint price over the next few trading days**.

## Main Results

The project reached four main conclusions:

1. **Bid/ask variables dominate the level relationship.** Both bid and ask coefficients were estimated very close to 1, which is expected because these quantities are mechanically closely related to the daily midpoint price.
2. **Plain regression was insufficient.** The original OLS residuals showed highly significant autocorrelation under the Ljung-Box test.
3. **ARIMA error modeling materially improved diagnostics.** Ljung-Box p-values rose to 0.4642 and 0.4721 for the two final models.
4. **The ask-based specification was slightly preferred.** Model 2.1 had lower AIC and BIC than Model 1.1.

## Limitations

The project itself identifies several important limitations:

- cryptocurrency prices are highly volatile
- external events and regulation are not included
- the historical sample is relatively short
- long-run seasonality and broader market cycles are difficult to identify
- bid and ask prices are closely related to the response itself, so coefficient interpretation should be treated carefully

Accordingly, the analysis is best viewed as a statistical modeling exercise rather than a production trading model.

## My Contribution

This was a three-person STAT 429 team project at the University of Illinois Urbana-Champaign. My documented responsibilities included:

- modeling the specification using **time + trading volume + selling price (`ask`)**
- contributing to presentation and slide design

## Repository Contents

- `R code.Rmd` — full R analysis
- `ZRXUSD.csv` — project dataset
- `STAT 429 Proposal.pdf` — project proposal
- `STAT429 UG3.pdf` — final project material

## Project Presentation

[Watch the project presentation](https://youtu.be/upVRhvnteMk)

## Tools

R · `forecast` · `astsa` · `tseries` · `ggplot2` · regression · ARIMA · residual diagnostics · forecasting

## Note

This repository is retained as an academic portfolio project. Results reflect the original course-project scope and assumptions and should not be interpreted as investment advice.
