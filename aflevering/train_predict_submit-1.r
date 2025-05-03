library(data.table)

# Modify this variable to your group name, which is formatted as FirstName1_LastName1-FirstName2_LastName2-...
GROUP_NAME <- "Kasper_Jensen-Jakob_Nielsen"

# Define the train function, assume that X is a data.table and y is a vector of responses of the same length as the number of rows in X
train_model <- function(X, y) {
  # Do your training here, do hyperparameter tuning, etc.
  # The following code is an example of how to train a simple xgboost model in mlr3

  # ---- MODIFY THIS SECTION ----
  library(mlr3verse)
  library(mlr3)
  library(mlr3learners)
  library(mlr3tuning)
  library(mlr3pipelines)

  task <- as_task_regr(data.table(X, y = y), target = "y")

  # Select only numeric columns for training
  select_num <- po("select", selector = selector_type(c("numeric", "integer")))

  learner <- select_num %>>%
    lrn("regr.xgboost",
      nrounds = to_tune(10, 100),
      max_depth = to_tune(1, 10),
      eta = to_tune(0.01, 0.1)
    )


  at <- auto_tuner(
    tuner = tnr("grid_search"),
    learner = learner,
    resampling = rsmp("cv", folds = 5),
    measure = msr("regr.mse"),
    term_evals = 10
  )
  at$train(task)
  model <- at$learner

  # ---- END OF MODIFY THIS SECTION ----

  predict_fun <- function(test_X) {
    # assume that test_X is the same format as X
    # call your model here, e.g. in mlr3 you can use the predict function
    # ---- MODIFY THIS SECTION ----
    preds <- model$predict_newdata(test_X)
    return(preds$response)
    # ---- END OF MODIFY THIS SECTION ----
  }

  return(predict_fun)
}

# run the following function to check if your train_model function is working
# do not modify this function
test_model_passes <- function() {
  data <- fread("Motor vehicle insurance data.csv", sep = ";")

  X <- data[, -c("Cost_claims_year")]
  y <- data$Cost_claims_year

  train_idx <- sample(seq_along(y), 10000)
  test_idx <- sample(setdiff(seq_along(y), train_idx), 10000)

  train_X <- X[train_idx]
  train_y <- y[train_idx]
  test_X <- X[test_idx]
  test_y <- y[test_idx]

  predict_fun <- train_model(train_X, train_y)

  pred <- predict_fun(test_X)
  pred_baseline <- mean(train_y)
  MSE <- mean((pred - test_y)^2)
  baseline_MSE <- mean((pred_baseline - test_y)^2)

  print(paste("MSE:", MSE))
  print(paste("Baseline MSE:", baseline_MSE))
}

# uncomment this line to test your model
# test_model_passes()
