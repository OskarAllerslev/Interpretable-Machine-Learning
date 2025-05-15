library(data.table)

# Modify this variable to your group name, which is formatted as FirstName1_LastName1-FirstName2_LastName2-...
GROUP_NAME <- "Kasper_Jensen-Jakob_Nielsen"

# Define the train function, assume that X is a data.table and y is a vector of responses of the same length as the number of rows in X
train_model <- function(X, y) {
  # Do your training here, do hyperparameter tuning, etc.
  # The following code is an example of how to train a simple xgboost model in mlr3

  # ---- MODIFY THIS SECTION ----
  library(mice)
  library(mlr3)
  library(mlr3learners )
  library(mlr3tuning)
  library(mlr3mbo)
  library(glmnet)
  library(OpenML)
  library(mlr3pipelines)
  library(future)
  library(magrittr)
  library(data.table)
  library(tidyverse)
  library(corrr)
  library(corrplot)
  library(mlr3verse)
  data_trans <- function(data){
    
    today <- Sys.Date()
    
    date_cols <- c("Date_start_contract", 
                   "Date_last_renewal",
                   "Date_next_renewal", 
                   "Date_lapse")
    data[, (date_cols) := lapply(.SD, as.IDate, format = "%d/%m/%Y"),
         .SDcols = date_cols]
    data[, Period_end :=
           fifelse(
             !is.na(Date_lapse) &
               Date_lapse >= Date_last_renewal &            
               Date_lapse <= Date_next_renewal,            
             Date_lapse,
             pmin(Date_next_renewal, today, na.rm = TRUE) 
           )]
    
    
    data[, Exposure_days  := as.numeric(Period_end - Date_last_renewal)]
    data[, Exposure_days  := pmax(Exposure_days, 0)]      
    data[, Exposure_unit  := fcase(
      between(Exposure_days, 364, 367), 1,
      default = Exposure_days/365.25
    )]
    
    
    #data[, Cost_claims_sum := sum(Cost_claims_year), by = ID]
    
    # Add a column that flags the latest row per group
    
    setorder(data, ID, -Date_last_renewal)
    data[, is_second_latest := FALSE]
    data[, is_second_latest := .I == .I[2], by = ID]
    # data$Cost_claims_year <- NA
    
    key_vars <- c("ID","Year_matriculation","Power",
                  "Cylinder_capacity","N_doors","Type_fuel","Weight")
    
    agg <- data[, .(
      Exposure               = sum(Exposure_unit),
      Max_policies_max       = max(Max_policies),
      Policies_in_force_second_latest = Policies_in_force[is_second_latest],
      
      Max_products_max       = max(Max_products),
      Payment_max            = max(Payment),
      Premium_second_latest = Premium[is_second_latest],
      Cost_claim_this_year = Cost_claims_year[which.max(Date_last_renewal)],
      Cost_claims_sum_history = sum(Cost_claims_year) -(Cost_claims_year[which.max(Date_last_renewal)]),
      R_claims_history   = max(N_claims_history - N_claims_year)/sum(Exposure_unit),
      # N_claims_year_sum      = sum(N_claims_year),
      N_claims_history   = max(N_claims_history - N_claims_year),
      # R_claims_history_sum   = sum(R_Claims_history),
      Type_risk_max          = max(Type_risk),
      Value_vehicle_mean     = mean(Value_vehicle),
      N_doors_mean           = mean(N_doors),
      Length_sum             = sum(Length, na.rm = TRUE)
    ), by = key_vars]
    
    
    
    
    
    agg <- agg %>% 
      dplyr::mutate(claim_indicator = dplyr::if_else(Cost_claims_sum_history + Cost_claim_this_year > 0, 1, 0))
    
    tmp <- agg %>%
      dplyr::filter(is.na(Type_fuel))
    
    NA_fuel_Y <- sum(tmp$Cost_claims_sum_history + tmp$Cost_claims_this_year)
    total_Y <- sum(agg$Cost_claims_sum_history + agg$Cost_claims_this_year)
    
    na_Y_ratio <- NA_fuel_Y / total_Y
    
    agg <- agg %>% 
      dplyr::filter(!is.na(Type_fuel))
    
    agg <- agg %>%
      dplyr::mutate(Type_fuel = dplyr::if_else(Type_fuel == "P", 1, 0))
    return(agg)
  }
  ### flet data igen
  dat <- X %>% dplyr::mutate(Cost_claims_year = y) 
  dat <- data_trans(dat) 
  
  
  # funk til frekvens -------------------
  train_freq <- function(data, folds, n_evals){
    
    data <- data %>% dplyr::select(-Cost_claim_this_year)
    
    task_freq = as_task_classif(
      data,
      target = "claim_indicator",
      positive = "1",
      weights = data$Exposure,
      id     = "frek_binary"
    )
  task_freq$set_col_roles("Exposure", "weight")
  
  graph_freq  = po("encode") %>>%
    po("scale")  %>>%
    lrn("classif.xgboost",
        predict_type = "prob",
        eval_metric  = "logloss", #Bedre loss funktion?
        nrounds      = to_tune(200, 800),
        max_depth    = to_tune(3, 7),
        eta          = to_tune(0.01, 0.3),
        subsample    = to_tune(0.6, 1))
  
  at_freq = auto_tuner(
    learner    = as_learner(graph_freq),
    resampling = rsmp("cv", folds = folds), #ÆNDRET 5 -> 2
    measure    = msr("classif.bbrier"),
    tuner      = tnr("random_search"),
    term_evals = n_evals #Ændret 5 til 1
  )
  model_freq <- at_freq$train(task_freq)
    return(model_freq)
  }
  
  
  # funk til skader --------------- 
  train_severity <- function(data, folds, n_evals){
    data <- data %>% dplyr::select(-claim_indicator)
    data <- data %>% dplyr::filter(Cost_claim_this_year > 0)
    
    task_S = as_task_regr(
      data,
      target = "Cost_claim_this_year",
      weights = data$Exposure,
      id     = "skade"
    )
    
    task_S$set_col_roles("Exposure", "weight")
    
    graph_S = po("encode") %>>%
      #po("scale") %>>%
      po("learner", lrn("regr.ranger"))
    
    glrn_S = GraphLearner$new(graph_S)
    
    at_severity = AutoTuner$new(
      learner    = glrn_S,
      resampling = rsmp("cv", folds = folds),  
      measure    = msr("regr.mse"),
      tuner      = tnr("random_search"),
      terminator = trm("evals", n_evals = n_evals)  
    )
    
    
    at_severity <- at_severity$train(task_S)
    return(at_severity) 
  }
  model_freq <<- train_freq(dat, folds = 2, n_evals = 2)
  model_severity <<- train_severity(dat, folds = 2, n_evals = 2)
  
  

  

  # ---- END OF MODIFY THIS SECTION ----

  predict_fun <- function(test_X) {
    # assume that test_X is the same format as X
    # call your model here, e.g. in mlr3 you can use the predict function
    # ---- MODIFY THIS SECTION ----
    test_X$Cost_claims_year <- 0 
    # jeg er ldit i tvivlt om det her er helt lovligt 
    test_X <- data_trans(test_X)
    browser()
    freq_pred <- model_freq$predict_newdata(test_X)
    severity_pred <- model_severity$predict_newdata(test_X)
    
    combined_pred <- freq_pred$prob[, 1] * severity_pred$response
    
    return(combined_pred)
    # ---- END OF MODIFY THIS SECTION ----
  }

  return(predict_fun)
}

# run the following function to check if your train_model function is working
# do not modify this function
test_model_passes <- function() {
  # data <- fread("Motor vehicle insurance data.csv", sep = ";")
  data <- fread("aflevering/Motor vehicle insurance data.csv", sep = ";")

  X <- data[, -c("Cost_claims_year")]
  y <- data$Cost_claims_year

  train_idx <- sample(seq_along(y), 10000)
  test_idx <- sample(setdiff(seq_along(y), train_idx), 10000)

  train_X <- X[train_idx]
  train_y <- y[train_idx]
  test_X <- X[test_idx]
  test_y <- y[test_idx]
  browser()

  predict_fun <- train_model(train_X, train_y)

  pred <- predict_fun(test_X)
  pred_baseline <- mean(train_y)
  MSE <- mean((pred - test_y)^2)
  baseline_MSE <- mean((pred_baseline - test_y)^2)

  print(paste("MSE:", MSE))
  print(paste("Baseline MSE:", baseline_MSE))
}

# uncomment this line to test your model
test_model_passes()
