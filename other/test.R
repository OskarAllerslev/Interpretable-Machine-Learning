library(mlr3verse)
library(quantmod)
library(tidyverse)
library(mlr3temporal)
library(mlr3extralearners)
library(data.table)


getSymbols("F", from = "2024-04-23")
dt <- tibble(date = index(F),
             ret  = dailyReturn(Cl(F), type = "log")) %>%
       mutate(ret_lag1 = lag(ret),
              ma20     = rollmean(ret, 20, fill = NA, align = "right"))
dt <- na.omit(dt)
task <- TaskRegrForecast$new(
  id       = "ford_logret",
  backend  = dt,
  target   = "ret",
  date_col = "date"
)

learner <- lrn("forecast.auto_arima")
rsmp <- rsmp("forecast_cv", folds = 5, window_size = 100, horizon = 5)

rr <- resample(task, learner, rsmp)
rr$aggregate(msr("forecast.rmse"))

preds <- rr$predictions()






all_preds <- rbindlist(
  lapply(preds, function(p) {
    d <- as.data.table(p)
    # Hent eksplicit kolonne 2 og 3 – uanset at de hedder det samme
    actual   <- d[[2]]
    forecast <- d[[3]]
    data.table(row_ids = d$row_ids, actual = actual, forecast = forecast)
  }))






ggplot(all_preds, aes(x = row_ids)) +
  geom_line(aes(y = actual), color = "black") +
  geom_line(aes(y = forecast), color = "blue") +
  labs(title = "AutoARIMA: Actual vs Forecast",
       x = "Observation", y = "Log-afkast") +
  theme_minimal()






all_preds[, position := fifelse(forecast > 0, 1, -1)]
all_preds[, return := actual * shift(position, fill = 0)]
all_preds[, cum_pnl := cumsum(return)]

ggplot(all_preds, aes(x = row_ids)) +
  geom_line(aes(y = cum_pnl), color = "darkgreen") +
  labs(title = "Simuleret PnL for Forecast-baseret strategi",
       x = "Observation", y = "Kumulativ log-afkast") +
  theme_minimal()





