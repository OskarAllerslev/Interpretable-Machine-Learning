
# load packages -----------------------------------------------------------
library(ggplot2)
library(dplyr)
library(mlr3)
library(mlr3learners)
library(OpenML)
library(mlr3pipelines)
library(mlr3verse)


# fetch and edit data -----------------------------------------------------

bike_data = getOMLDataSet(data.id = 42713)
bike_data$data <- bike_data$data[,-c(7,13,14)] ## remove casual and registered as sum of the two is count. also remove working day due to collinearity.

### convert dates to factors 
bike_data$data$year <- factor(bike_data$data$year)
bike_data$data$month <- factor(bike_data$data$month)
bike_data$data$hour <- factor(bike_data$data$hour)
bike_data$data$weekday <- factor(bike_data$data$weekday)



# a -----------------------------------------------------------------------
# run simple LS reg with Y ount and predictor windspeed


simple_ls_model <- stats::lm(count ~ windspeed, data = bike_data )
print(coef(simple_ls_model))
cat("The intercept of the model is: ", coef(simple_ls_model)[1], " and the windspeed coefficient is: ", coef(simple_ls_model)[2])




# b ls with all remaining predictors --------------------------------------

dat <- bike_data$data


task <- TaskRegr$new(id = "dat", backend = dat, target = "count")
po_encode <- po("encode")
learner = lrn("regr.lm")
graph = po_encode %>>% learner 
graph_learner = GraphLearner$new(graph)

# train the model

graph_learner$train(task)
coefficients = graph_learner$model$regr.lm$model$coefficients
cat("The windspeed coef is: ", coefficients["windspeed"])
resids <- graph_learner$model$regr.lm$model$residuals


# c -----------------------------------------------------------------------
#Do the following steps
#Run least squares linear regression with response being count and predictor being all remaining variables except windspeed. Calculate the residuals and call that variable count_residuals.
#Run least squares linear regression with response being windspeed and predictor being all remaining variables except count. Calculate the residuals and call that variable windpseed_residuals.
#Run simple least squares linear regression with response being count_residuals and predictor windpseed_residuals.
#Report the regression coefficient you get.

dat_no_wind = dat %>% dplyr::select(-windspeed) 
model1 <- stats::lm(count ~ . , data = dat_no_wind)
count_residuals = residuals(model1)

dat_no_count = dat %>% dplyr::select(-count)

model2 <- stats::lm(windspeed ~ ., data = dat_no_count )
windspeed_residuals <- residuals(model2)


model3 <- stats::lm(count_residuals ~ windspeed_residuals, data = data.frame(count_residuals, windspeed_residuals))
cat("The residuals of the residual lm model is: ", summary(model3)$coefficients)
plot(model3)
model3_residuals <- residuals(model3)



# d verify that coefficients are the same ---------------------------------


# we want to verify that the residuals from model 2 and model 3 are the same
coef_b = coefficients["windspeed"]
coef_c = summary(model3)$coefficients["windspeed_residuals", "Estimate"]
# they are the same



# e replae simpel lm with auto tuned k nearest neighnors  -----------------

residual_data <- data.frame(
  count_residuals = count_residuals,
  windspeed_residuals = windspeed_residuals
)

task_resid = TaskRegr$new(id = "residual_task", 
                          backend = residual_data, 
                          target = "count_residuals")
# make learner for KNN

learner_knn = lrn("regr.kknn")

# define tuning for k from lets say 1 to 50 
search_space = paradox::ps(k = paradox::p_int(lower = 1, upper = 50))

# resampling for tuning
resampling = rsmp("holdout", ratio = 0.9)

# measure residual mean square error 
measure = msr("regr.rmse")

# tuner 
tuner = mlr3tuning::tnr("random_search")

at = mlr3tuning::AutoTuner$new(
  learner = learner_knn,
  resampling = resampling,
  measure = measure,
  search_space = search_space,
  terminator = trm("evals", n_evals = 30),
  tuner = tuner
)


at$train(task_resid)
predictions = at$predict(task_resid)

plot_data <- data.frame(
  windspeed_residuals = windspeed_residuals,
  observed = count_residuals,
  predicted = predictions$response
)


ggplot2::ggplot(plot_data, ggplot2::aes(x = windspeed_residuals)) +
  geom_point(aes(y = observed), color = "blue", alpha = 0.5) +
  geom_line(aes(y = predicted), color = "darkred")

# it becomes clear here, that kNN captures non linear patterns better than the linear model
# the normal linear model was okay, the residuals were nice enough. 












