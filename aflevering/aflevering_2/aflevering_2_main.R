

# pakker ------------------------------------------------------------------
library(ALEPlot)
library(fastshap)
library(iml)
library(DALEX)
library(rsample)
library(DiagrammeR)
library(gt)
library(skimr)
library(patchwork)
library(mlr3viz)
library(GGally)
library(tweedie)
library(mlr3extralearners)
library(Hmisc)
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
library(xgboost)
library(glex)




# hjælpefunktioner --------------------------------------------------------



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
  
  
  key_vars <- c("ID","Year_matriculation","Power",
                "Cylinder_capacity","N_doors","Type_fuel","Weight", "Length",
                "Area", "Value_vehicle", "Date_birth", "Date_driving_licence", "Date_start_contract")
  
  data$Date_birth <- as.integer(
    base::format(
      base::as.Date(data$Date_birth, format = "%d/%m/%Y"),
      "%Y"
    )
  )
  
  data$Date_driving_licence <- as.integer(
    base::format(
      base::as.Date(data$Date_driving_licence, format = "%d/%m/%Y"),
      "%Y"
    )
  )
  
  
  data$Date_start_contract <- as.integer(
    base::format(
      base::as.Date(data$Date_start_contract, format = "%d/%m/%Y"),
      "%Y"
    )
  )
  
  
  
  
  data <- data %>% dplyr::filter(Exposure_unit != 0)
  agg <- data[, .(
    Exposure               = sum(Exposure_unit),
    Max_policies_max       = max(Max_policies),
    Max_lapse              = max(Lapse),
    dist_channel_0 = as.integer(any(Distribution_channel == "0")),
    dist_channel_1 = as.integer(any(Distribution_channel == "1")),
    dist_channel_2 = as.integer(any(!(Distribution_channel %in% c("0", "1")))),
    Max_products_max       = max(Max_products),
    Payment_max            = max(Payment),
    Mean_premium            = (sum(Premium) - Premium[which.max(Date_last_renewal)]) / (sum(Exposure_unit)),
    Cost_claim_this_year = Cost_claims_year[which.max(Date_last_renewal)],
    Cost_claims_sum_history = sum(Cost_claims_year) -(Cost_claims_year[which.max(Date_last_renewal)]),
    R_claims_history   = max(N_claims_history - N_claims_year)/sum(Exposure_unit),
    N_claims_history   = max(N_claims_history - N_claims_year),
    Type_risk_max          = max(Type_risk),
    Value_vehicle_mean     = mean(Value_vehicle)
  ), by = key_vars]
  
  
  agg <- agg %>%
    dplyr::mutate(claim_indicator = dplyr::if_else(Cost_claims_sum_history + Cost_claim_this_year > 0, 1, 0))
  
  agg <- agg %>%
    dplyr::mutate(Type_fuel = dplyr::if_else(Type_fuel == "P", 1, 0))
  
  length_model <- lm(
    Length ~ Cylinder_capacity + Weight + Value_vehicle,
    data = na.omit(agg)
  )
  fuel_model <- lm(
    Type_fuel ~ Cylinder_capacity + Weight + Value_vehicle,
    data = na.omit(agg)
  )
  
  agg <- agg %>%
    dplyr::mutate(
      pred_length = stats::predict(length_model, newdata = .),
      pred_type_fuel = stats::predict(fuel_model, newdata = .)
    ) %>%
    dplyr::mutate(
      Length = dplyr::if_else(is.na(Length), pred_length, Length),
      Type_fuel = dplyr::if_else(is.na(Type_fuel), pred_type_fuel, Type_fuel)
    ) %>%
    dplyr::select(-pred_length, -pred_type_fuel)
  
  
  return(agg)
}





# gammel split model ------------------------------------------------------
data <- fread("aflevering/Motor vehicle insurance data.csv", sep = ";")
data_t <- data_trans(data)
## frek model  ----
data_F <- data_t
data_F <- data_F %>%
  dplyr::select(-Cost_claim_this_year, - ID)



task_freq <- as_task_classif(
  data_F,
  target = "claim_indicator",
  positive = "1",
  weights = "Exposure",
  id = "frequency"
)


task_freq$set_col_roles("Exposure", "weight")

# stratificer tjek at det faktisk er hvad der sker her
task_freq$col_roles$stratum = "claim_indicator"


prep_graph = po("encode") %>>%
  po("scale")

g_featureless_f =
  prep_graph %>>%
  po("learner", lrn("classif.featureless", predict_type = "prob"))

g_rf_f = po("encode") %>>%
  po("learner", lrn("classif.ranger", predict_type = "prob",
                    num.trees = to_tune(100L, 500L),
                    mtry = to_tune(1L, as.integer(sqrt(ncol(data_F)-2)))))


g_lgbm_f =
  prep_graph %>>%
  po("learner",
     lrn("classif.lightgbm",
         objective              = "binary",
         learning_rate          = to_tune(1e-3, 0.2, logscale = TRUE),
         num_leaves             = to_tune(16L, 32L),
         num_iterations         = to_tune(200L, 1000L))
  )

g_xgb_f =
  prep_graph %>>%
  po("learner",
     lrn("classif.xgboost",
         predict_type = "prob",
         eval_metric = "logloss",
         nrounds      = to_tune(200, 800),
         max_depth    = to_tune(3, 7),
         eta          = to_tune(0.01, 0.3),
         subsample    = to_tune(0.6, 1)
     ))

g_ranger_f <- prep_graph %>>%
  po("learner",
     lrn("regr.ranger",
         mtry = 5,
         min.node.size = 1, 
         importance = "permutation"
     )
  )

g_enet_f =
  prep_graph %>>%
  po("learner",
     lrn("classif.glmnet", predict_type = "prob",
         alpha = to_tune(0, 1),
         s     = to_tune(1e-4, 1, logscale = TRUE))
  )


auto = function(graph, id)
{
  at = AutoTuner$new(
    learner = GraphLearner$new(graph, id = id),
    resampling = rsmp("cv", folds = 2),
    measure = msr("classif.bbrier"),
    tuner = tnr("random_search"),
    terminator = trm("evals", n_evals = 1)
  )
  invisible(at)
}

at_lgbm_f = auto(g_lgbm_f, "lgbm")
at_xgb_f = auto(g_xgb_f, "xgb")
at_enet_f = auto(g_enet_f, "enet")
at_rf_f = auto(g_rf_f, "ranger")
at_featureless_f = auto(g_featureless_f, "featureless")


tasks_f = list(task_freq)
learners_f = list(at_lgbm_f, at_xgb_f, at_enet_f, at_featureless_f, at_rf_f)
design_f = benchmark_grid(task = tasks_f, learners = learners_f,
                          resampling = rsmp("cv", folds = 2))

kør_igen <- "nej"


if (!file.exists("~/Interpretable-Machine-Learning/aflevering/benchmark_frekvens.rds" ) | kør_igen == "ja") {
  bmr_f_gammel = benchmark(design_f, store_models = TRUE)
  saveRDS(bmr_f_gammel, "~/Interpretable-Machine-Learning/aflevering/benchmark_frekvens.rds")
} else {
  bmr_f_gammel = readRDS("~/Interpretable-Machine-Learning/aflevering/benchmark_frekvens.rds")
}


## sev model ---- 


data_S <- data_t
data_S <- data_S %>% dplyr::select(-claim_indicator, -ID)
data_S <- data_S %>% dplyr::filter(Cost_claim_this_year > 0)
data_S <- data_S %>% mutate(Cost_claim_this_year = log(Cost_claim_this_year))

task_S = as_task_regr(
  data_S,
  target = "Cost_claim_this_year",
  weights = "Exposure",
  id     = "severity"
)

task_S$set_col_roles("Exposure", "weight")


prep_graph = po("encode") %>>%
  po("scale")

g_featureless_s =
  prep_graph %>>%
  po("learner",
     lrn("regr.featureless")
  )



g_lgbm_s =
  prep_graph %>>%
  po("learner",
     lrn("regr.lightgbm",
         objective     = "regression",
         #metric        = "rmse",
         learning_rate          = to_tune(1e-3, 0.2, logscale = TRUE),
         num_leaves             = to_tune(16L, 32L),
         num_iterations         = to_tune(200L, 1000L))
  )


g_xgb_s =
  prep_graph %>>%
  po("learner",
     lrn("regr.xgboost",
         objective = "reg:squarederror",
         eta                    = to_tune(1e-3, 0.2, logscale = TRUE),
         max_depth              = to_tune(3L, 9L),
         nrounds                = to_tune(200L, 1000L))
  )

g_enet_s =
  prep_graph %>>%
  po("learner",
     lrn("regr.glmnet",
         alpha = to_tune(0, 1),
         s     = to_tune(1e-4, 1, logscale = TRUE))
  )

g_ranger_s <- prep_graph %>>%
  po("learner",
     lrn("regr.ranger",
         mtry = 5,
         min.node.size = 5,  
         importance = "permutation"
     )
  )

auto = function(graph, id)
{
  at_s = AutoTuner$new(
    learner = GraphLearner$new(graph, id = id),
    resampling = rsmp("cv", folds = 5),
    measure = msr("regr.mse"),
    tuner = tnr("random_search"),
    terminator = trm("evals", n_evals = 1)
  )
  invisible(at_s)
}

at_lgbm_s = auto(g_lgbm_s, "lgbm")
at_xgb_s = auto(g_xgb_s, "xgb")
at_enet_s = auto(g_enet_s, "enet")
at_featureless_s = auto(g_featureless_s, "featureless")
at_ranger_s = auto(g_ranger_s, "ranger")

tasks_s = list(task_S)
learners_s = list(at_lgbm_s, at_xgb_s, at_enet_s,at_ranger_s, at_featureless_s)
#learners_s = list(at_ranger, at_featureless_s)
design_s = benchmark_grid(task = tasks_s, learners = learners_s,
                          resampling = rsmp("cv", folds = 5))

kør_igen <- "nej"


if (!file.exists("~/Interpretable-Machine-Learning/aflevering/benchmark_severity.rds" ) | kør_igen == "ja") {
  bmr_s_gammel = benchmark(design_s, store_models = TRUE)
  saveRDS(bmr_s_gammel, "~/Interpretable-Machine-Learning/aflevering/benchmark_severity.rds")
} else {
  bmr_s_gammel = readRDS("~/Interpretable-Machine-Learning/aflevering/benchmark_severity.rds")
}

# fortolkelig split model -------------------------------------------------

## frequency model interpretable ----


g_xgb_f_interpretable =
  prep_graph %>>%
  po("learner",
     lrn("classif.xgboost",
         predict_type = "prob",
         eval_metric   = "logloss",
         max_depth     = 2,      
         eta           = to_tune(0.05, 0.1),
         nrounds       = to_tune(50, 150),
         subsample     = to_tune(0.6, 1)
     )
  )


at_xgb_f_interp = AutoTuner$new(
  learner   = GraphLearner$new(g_xgb_f_interpretable, id = "xgb_interp"),
  resampling = rsmp("cv", folds = 2),
  measure    = msr("classif.bbrier"),          
  tuner      = tnr("random_search"),
  terminator = trm("evals", n_evals = 2),    
  store_tuning_instance = TRUE               
)

outer_rsmp_f = rsmp("cv", folds = 2)            

kør_igen <- "nej"

if (!file.exists("~/Interpretable-Machine-Learning/aflevering/inte_pretable_frekvens.rds" ) | kør_igen == "ja") {
  rr_freq_interp = resample(
    task       = task_freq,
    learner    = at_xgb_f_interp,
    resampling = outer_rsmp_f,
    store_models = TRUE                          
  )
  saveRDS(rr_freq_interp, "~/Interpretable-Machine-Learning/aflevering/inte_pretable_frekvens.rds")
} else {
  rr_freq_interp = readRDS("~/Interpretable-Machine-Learning/aflevering/inte_pretable_frekvens.rds")
}



classif_bbrier <- rr_freq_interp$aggregate(msr("classif.bbrier"))


p1 <- autoplot(
  rr_freq_interp, 
  measure = msr("classif.bbrier"),
  type = "boxplot"
) + 
  ggplot2::ggtitle("Frequency")




measure_f <- msr("classif.bbrier")

scores_f <- rr_freq_interp$score(measure_f)            # data.table
best_fold_f <- scores_f[, .I[ which.min(classif.bbrier) ] ]
best_at_f <- rr_freq_interp$learners[[ best_fold_f ]]

best_params_f <- as.data.table(best_at_f$tuning_result$x)
print(best_params_f)

best_score_f <- scores_f[best_fold_f, classif.bbrier]





## severity model interpretable ----
g_xgb_s_interpretable <- 
  prep_graph %>>%
  po("learner",
     lrn("regr.xgboost",
         max_depth      = 2,          
         eta           = to_tune(0.05, 0.1),
         nrounds       = to_tune(50, 150),
         subsample     = to_tune(0.6, 1)   ,    
         colsample_bytree = to_tune(0.7, 1),     
         min_child_weight = to_tune(1, 10),       
         booster        = "gbtree",
         objective      = "reg:squarederror",
         tree_method    = "hist"  
     )
  )


at_xgb_s_interp = AutoTuner$new(
  learner   = GraphLearner$new(g_xgb_s_interpretable, id = "xgb_interp"),
  resampling = rsmp("cv", folds = 2),
  measure    = msr("regr.mse"),                
  tuner      = tnr("random_search"),
  terminator = trm("evals", n_evals = 2),
  store_tuning_instance = TRUE
)

outer_rsmp_s = rsmp("cv", folds = 2)



kør_igen <- "nej"

if (!file.exists("~/Interpretable-Machine-Learning/aflevering/inte_pretable_severity_xgb.rds" ) | kør_igen == "ja") {
  rr_sev_interp = resample(
    task         = task_S,
    learner      = at_xgb_s_interp,
    resampling   = outer_rsmp_s,
    store_models = TRUE
  )
  saveRDS(rr_sev_interp, "~/Interpretable-Machine-Learning/aflevering/inte_pretable_severity_xgb.rds")
} else {
  rr_sev_interp = readRDS("~/Interpretable-Machine-Learning/aflevering/inte_pretable_severity_xgb.rds")
}



sev_mse <- rr_sev_interp$aggregate(msr("regr.mse"))

p2 <- autoplot(rr_sev_interp,
               measure = msr("regr.mse"),
               type = "boxplot") + 
  ggplot2::ggtitle("Severity")



measure_s <- msr("regr.mse")

scores_s   <- rr_sev_interp$score(measure_s)
best_fold_s <- scores_s[, .I[ which.min(regr.mse) ] ]
best_at_s   <- rr_sev_interp$learners[[ best_fold_s ]]

best_params_s <- as.data.table(best_at_s$tuning_result$x)
print(best_params_s)

best_score_s <- scores_s[best_fold_s, regr.mse]




cat("Sev log mse ",sev_mse, " classif bbrier ", classif_bbrier )  


(p1 + p2)  




# explain interpretable model ---------------------------------------------


best_freq_mod <- best_at_f$train(task_freq)
best_sev_mod <- best_at_s$train(task_S)
best_at_f$learner_model


X_freq <- as.data.frame(task_freq$data(cols = best_at_f$feature_names))
y_freq <- task_freq$truth()

X_sev  <- as.data.frame(task_S$data(cols = best_at_s$feature_names))
X_sev$Cost_claim_this_year <- NULL
y_sev  <- task_S$truth()





## shap-values local & global ----


#### shapley local for one person ----
predictor_freq <- Predictor$new(best_freq_mod,
                                data = X_freq,
                                y    = y_freq,
                                type = "prob")  

# udvalgt person
shap_freq <- Shapley$new(predictor_freq, x.interest = X_freq[1, ])
plot(shap_freq)                                


#### shaples for frek ----
# pfun <- function(object, newdata) {
#   pred <- object$predict_newdata(newdata)  
#   as.numeric(pred$prob[, 2])
# }
# 
# 
# sh_freq <- fastshap::explain(
#   object       = best_freq_mod,
#   X            = as.data.frame(X_freq), 
#   pred_wrapper = pfun,
#   nsim         = 500,                    
#   adjust       = TRUE                     
# )




#### shapley local for one person ----

predictor_sev <- Predictor$new(best_sev_mod,
                               data = X_sev,
                               y    = y_sev)

shap_sev <- Shapley$new(predictor_sev, x.interest = X_sev[1, ])
plot(shap_sev)


######################################## frek ale 
best_par <- as.list(best_at_f$tuning_result$x)

final_xgb <- lrn(
  "classif.xgboost",
  predict_type = "prob",
  eval_metric  = "logloss",
  max_depth    = 2,                                   
  eta          = best_par[[1]]$classif.xgboost.eta,
  nrounds      = best_par[[1]]$classif.xgboost.nrounds,
  subsample    = best_par[[1]]$classif.xgboost.subsample
)

final_xgb$train(task_freq)





X_df <- task_freq$data(cols = final_xgb$feature_names) |> as.data.frame()

j <- which(names(X_df) == "Weight")  



pred_freq <- function(X.model, newdata) {
  X.model$predict_newdata(newdata)$prob[, "1"]
}

ALEPlot(
  X        = X_df,
  X.model  = final_xgb,      
  pred.fun = pred_freq,
  J        = j,
  K        = 40
)


# i could not get it to work with the categorical vars
num_vars <- names(X_df)[sapply(X_df, is.numeric)]
num_vars <- setdiff(num_vars, "cost_claim_this_year")
ale_long <- map_dfr(
  num_vars,
  function(v) {
    j   <- which(names(X_df) == v)
    tmp <- tempfile(); png(tmp)        
    ale <- ALEPlot(X_df, final_xgb, pred_freq, J = j, K = 40, NA.plot = TRUE)
    dev.off(); unlink(tmp)
    if (sd(ale$f.values) < 1e-6) return(NULL)
    
    tibble::tibble(
      variable = v,
      x        = ale$x.values,        
      ale      = ale$f.values        
    )
  }
)


ggplot(ale_long, aes(x, ale)) +
  geom_line(linewidth = .6) +
  facet_wrap(~ variable, scales = "free_x") +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = .3) +
  scale_y_continuous(breaks = seq(-.4,.4,.2)) +
  labs(y = "ALE", x = NULL,
       title = "ALE-plots") +
  theme_bw() +
  theme(strip.text = element_text(size = 8),
        panel.grid.major.y = element_line(colour = "grey85"),
        panel.grid.minor.y = element_blank())

########################################## sev ale 



best_par_s <- as.list(best_at_s$tuning_result$x)

final_xgb_s <- lrn(
  "regr.xgboost",
  predict_type        = "response",
  nrounds             = best_par_s[[1]]$regr.xgboost.nrounds,
  max_depth           = 2,
  eta                 = best_par_s[[1]]$regr.xgboost.eta,
  subsample           = best_par_s[[1]]$regr.xgboost.subsample,
  colsample_bytree    = best_par_s[[1]]$regr.xgboost.colsample_bytree,
  min_child_weight    = best_par_s[[1]]$regr.xgboost.min_child_weight,
  booster             = "gbtree",
  objective           = "reg:squarederror",
  tree_method         = "hist"
)

final_xgb_s$train(task_S)               

pred_sev <- function(X.model, newdata) {
  X.model$predict_newdata(newdata)$response   
}

X_df  <- task_S$data(cols = final_xgb_s$feature_names) |> as.data.frame()
num_vars <- names(X_df)[sapply(X_df, is.numeric)]

ale_long <- map_dfr(
  num_vars,
  function(v) {
    j   <- which(names(X_df) == v)
    tmp <- tempfile(); png(tmp)          
    ale <- ALEPlot(X_df, final_xgb_s, pred_sev,
                   J = j, K = 40, NA.plot = TRUE)
    dev.off(); unlink(tmp)
    if (sd(ale$f.values) < 1e-6) return(NULL)
    
    tibble(
      variable = v,
      x        = ale$x.values,
      ale      = ale$f.values
    )
  }
)

ggplot(ale_long, aes(x, ale)) +
  geom_line(linewidth = .6) +
  facet_wrap(~ variable, scales = "free_x") +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = .3) +
  labs(title = "ALE-plots",
       y = "ALE", x = NULL) +
  theme_bw() +
  theme(strip.text        = element_text(size = 8),
        panel.grid.major.y = element_line(colour = "grey85"),
        panel.grid.minor.y = element_blank())






# comparing the final model with the one from project 1 -------------------


rr_sev_interp$prediction()
rr_freq_interp$prediction()


DATA <- data_trans(data)
N <- nrow(DATA)
train_idx <- sample(seq_len(N), size = 0.8 * N)

train <- DATA[train_idx, ]
test <- DATA[-train_idx, ]

task_freq_train <- task_freq$clone()$filter(train_idx)
task_sev_train <- task_S$clone()$filter(train_idx)

lrn_freq_final <- best_at_f$clone(deep = TRUE)$train(task_freq_train)
lrn_sev_final <- best_at_s$clone(deep = TRUE)$train(task_sev_train)

E_X <- lrn_sev_final$predict_newdata(test)
E_N <- lrn_freq_final$predict_newdata(test)

res <- data.table::data.table(
  pred_N = E_N$prob[, 1],
  pred_X = exp(E_X$response)
)
res <- res %>% dplyr::mutate(pred_comb = pred_N * pred_X)

mse_combined <- mean((res$pred_comb - test$Cost_claim_this_year)^2)
mse_baseline <- mean((test$Cost_claim_this_year - mean(train$Cost_claim_this_year))^2)




# dat of birth stuff ------------------------------------------------------

#### nyt forsøg ----
debiased_pred <- function(glex_obj, vars_to_zero) {
  cols_rm <- grep(paste(vars_to_zero, collapse = "|"),
                  colnames(glex_obj$m), value = TRUE)
  m_tmp <- glex_obj$m
  m_tmp[, cols_rm] <- 0                
  rowSums(m_tmp) + glex_obj$intercept   
}

check_glex_identity <- function(glex_obj, model, X) {
  p1 <- rowSums(glex_obj$m) + glex_obj$intercept          
  p2 <- predict(model, X, outputmargin = TRUE)           
  c(max_abs  = max(abs(p1 - p2)),
    mean_abs = mean(abs(p1 - p2)))
}

feat <- task_freq$feature_names
df   <- as.data.frame(task_freq$data(cols = feat))
X    <- model.matrix(~ . - 1, df)

if ("positive" %in% levels(task_freq$truth())) {
  pos_lab <- "positive"
} else {
  pos_lab <- names(table(task_freq$truth()))[1]         
}
y <- as.integer(task_freq$truth() == pos_lab)

dtrain <- xgb.DMatrix(X, label = y)

params <- list(
  objective   = "binary:logistic",
  eval_metric = "logloss",
  eta         = bp$classif.xgboost.eta,
  max_depth   = 2L,
  subsample   = bp$classif.xgboost.subsample
)

xgb_cls <- xgb.train(params,
                     dtrain,
                     nrounds = bp$classif.xgboost.nrounds,
                     verbose = 0)

glex_cls <- glex(xgb_cls, X)
check_glex_identity(glex_cls, xgb_cls, X)

vars_rm <- c("Date_birth", "Max_Date_birth")

pred_org_margin <- predict(xgb_cls, X, outputmargin = TRUE)
pred_db_margin  <- debiased_pred(glex_cls, vars_rm)

pred_org_prob <- plogis(pred_org_margin)
pred_db_prob  <- plogis(pred_db_margin)

plot(pred_org_prob, pred_db_prob,
     xlab = "Original probability",
     ylab = "Debiased probability")
abline(0, 1, col = "red")

#### sev model plots ----

feat <- task_S$feature_names
df   <- as.data.frame(task_S$data(cols = feat))
X    <- model.matrix(~ . - 1, df)

y <- as.numeric(task_S$truth())
dtrain <- xgb.DMatrix(X, label = y)


bp_reg <- as.list(best_at_s$tuning_result$x)[[1]]  
names(bp_reg)                                     



params <- list(
  objective         = "reg:squarederror",
  eval_metric       = "rmse",
  eta               = bp_reg$regr.xgboost.eta,
  max_depth         = 2L,
  subsample         = bp_reg$regr.xgboost.subsample,
  colsample_bytree  = bp_reg$regr.xgboost.colsample_bytree,
  min_child_weight  = bp_reg$regr.xgboost.min_child_weight
)

nrounds <- as.integer(bp_reg$regr.xgboost.nrounds)

xgb_reg <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = nrounds,
  verbose = 0
)


glex_reg <- glex(xgb_reg, X)

debiased_pred <- function(glex_obj, vars_to_zero) {
  cols_rm <- grep(paste(vars_to_zero, collapse = "|"),
                  colnames(glex_obj$m), value = TRUE)
  m_tmp <- glex_obj$m
  m_tmp[, cols_rm] <- 0
  rowSums(m_tmp) + glex_obj$intercept     
}

vars_rm <- c("Date_birth", "Max_Date_birth")

pred_org <- predict(xgb_reg, X)                 
pred_db  <- debiased_pred(glex_reg, vars_rm)   
diff     <- pred_org - pred_db                


plot(exp(exp(pred_org)), exp(exp(pred_db)),
     xlab = "Original prediction",
     ylab = "De-biased prediction")
abline(0, 1, col = "red", lwd = 2)






# mse for debiased --------------------------------------------------------

feat_sev  <- task_S$feature_names     
feat_freq <- task_freq$feature_names
vars_rm   <- c("Date_birth", "Max_Date_birth")

missing_sev <- setdiff(feat_sev, colnames(test))

df_test_sev <- test %>%
  as_tibble() %>%
  {
    if (length(missing_sev) > 0) {
      for (col in missing_sev) {
        .[[col]] <- 0
      }
    }
    .
  } %>%
  select(all_of(feat_sev))

X_test_sev     <- model.matrix(~ . - 1, data = df_test_sev)
marg_org_sev   <- predict(xgb_reg, X_test_sev)
cost_org_sev   <- exp(exp(marg_org_sev))

glex_test_sev  <- glex(xgb_reg, X_test_sev)
marg_db_sev    <- debiased_pred(glex_test_sev, vars_rm)
cost_db_sev    <- exp(exp(marg_db_sev))

missing_clas <- setdiff(feat_freq, colnames(test))

df_test_clas <- test %>%
  as_tibble() %>%
  {
    if (length(missing_clas) > 0) {
      for (col in missing_clas) {
        .[[col]] <- 0
      }
    }
    .
  } %>%
  select(all_of(feat_freq))


X_test_freq    <- model.matrix(~ . - 1, data = df_test_clas)
marg_freq      <- predict(xgb_cls, X_test_freq, outputmargin = TRUE)
prob_freq      <- plogis(marg_freq)   # P(N>0)

pred_comb_org <- prob_freq * cost_org_sev
pred_comb_db  <- prob_freq * cost_db_sev
actual        <- test$Cost_claim_this_year

mse_org <- mean((pred_comb_org - actual)^2)
mse_db  <- mean((pred_comb_db  - actual)^2)
















#for classification
library(glex)



bp          <- as.list(best_at_f$tuning_result$x)[[1]]
eta         <- bp$classif.xgboost.eta
nrounds     <- bp$classif.xgboost.nrounds
subsample   <- bp$classif.xgboost.subsample
max_depth   <- 2L                          

# matrix-input
feat   <- task_freq$feature_names
df     <- as.data.frame(task_freq$data(cols = feat))
X      <- model.matrix(~ . - 1, data = df)
y      <- as.numeric(task_freq$truth())
dtrain <- xgb.DMatrix(X, label = y)


y <- ifelse(task_freq$truth() == "positive", 1L, 0L)
dtrain <- xgb.DMatrix(X, label = y)


params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta       = eta,
  max_depth = max_depth,
  subsample = subsample
)
xgb_class <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = nrounds,
  verbose = 0
)


glex_obj_F <- glex(xgb_class, X)


vars_rm  <- c("Date_birth", "Max_Date_birth")
cols_rm_F  <- grep(paste(vars_rm, collapse = "|"), colnames(glex_obj_F$m), value = TRUE)
m_db_F     <- glex_obj_F$m
m_db_F[, cols_rm_F] <- 0
pred_db_F  <- rowSums(m_db_F) + glex_obj_F$intercept


pred_org_F <- predict(xgb_class, X)



plot(pred_org_F, pred_db_F, xlab = "original", ylab = "debiased")
abline(0, 1)
diff <- (rowSums(glex_obj_F$m) + glex_obj_F$intercept) - pred_org_F
print(c(max = max(abs(diff)), mean = mean(abs(diff))))


#---------------------




# dat of birth stuff ------------------------------------------------------
#for regression



bp <- as.list(best_at_s$tuning_result$x)[[1]]

eta               <- bp$regr.xgboost.eta
nrounds           <- bp$regr.xgboost.nrounds
subsample         <- bp$regr.xgboost.subsample
max_depth         <- 2
colsample_bytree  <- bp$regr.xgboost.colsample_bytree
min_child_weight  <- bp$regr.xgboost.min_child_weight



# matrix-input
feat   <- task_S$feature_names
df     <- as.data.frame(task_S$data(cols = feat))
X      <- model.matrix(~ . - 1, data = df)
y      <- as.numeric(task_S$truth())
dtrain <- xgb.DMatrix(X, label = y)


params <- list(
  objective = "reg:squarederror",
  eta       = eta,
  max_depth = max_depth,
  subsample = subsample
)
xgb_reg <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = nrounds,
  verbose = 0
)


glex_obj <- glex(xgb_reg, X)

vars_rm  <- c("Date_birth", "Max_Date_birth")
cols_rm  <- grep(paste(vars_rm, collapse = "|"), colnames(glex_obj$m), value = TRUE)
m_db     <- glex_obj$m
m_db[, cols_rm] <- 0
pred_db  <- rowSums(m_db) + glex_obj$intercept


pred_org <- predict(xgb_reg, X)



plot(pred_org, pred_db, xlab = "original", ylab = "debiased")
abline(0, 1)
diff <- (rowSums(glex_obj$m) + glex_obj$intercept) - pred_org
print(c(max = max(abs(diff)), mean = mean(abs(diff))))


shap_birth_df <- data.frame(
  shap_value  = glex_obj$shap[["Date_birth"]],
  Date_birth  = df$Date_birth
)

head(shap_birth_df)

library(ggplot2)
ggplot(shap_birth_df, aes(x = Date_birth, y = shap_value)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  scale_y_continuous(limits = c(-0.05, 0.05)) +
  labs(title = "SHAP Value vs Date of Birth", y = "SHAP", x = "Date of Birth")

#-------------------------------------------------------------------------------

# MSE FINAL COMPARISON ----------------------------------------------------

# 
# rr_sev_interp$prediction()
# rr_freq_interp$prediction()
# 
# 
# DATA <- data_trans(data)
# N <- nrow(DATA)
# train_idx <- sample(seq_len(N), size = 0.8 * N)
# 
# train <- DATA[train_idx, ]
# test <- DATA[-train_idx, ]
# 
# task_freq_train <- task_freq$clone()$filter(train_idx)
# task_sev_train <- task_S$clone()$filter(train_idx)
# 
# lrn_freq_final <- best_at_f$clone(deep = TRUE)$train(task_freq_train)
# lrn_sev_final <- best_at_s$clone(deep = TRUE)$train(task_sev_train)
# 
# E_X <- lrn_sev_final$predict_newdata(test)
# E_X
# E_N <- lrn_freq_final$predict_newdata(test)
# E_N
# 
# res <- data.table::data.table(
#   pred_N = E_N$prob[, 1],
#   pred_X = exp(E_X$response)
# )
# res <- res %>% dplyr::mutate(pred_comb = pred_N * pred_X)
# 
# mse_combined <- mean((res$pred_comb - test$Cost_claim_this_year)^2)
# mse_baseline <- mean((test$Cost_claim_this_year - mean(train$Cost_claim_this_year))^2)
# 
# 
# 
# 
# m_db     <- glex_obj$m
# m_db[, cols_rm] <- 0
# pred_db  <- rowSums(m_db) + glex_obj$intercept
# pred_org <- predict(xgb_reg, X) 
# 
# MSE_S = (pred_db - as.numeric(task_S$truth()))^2
# 
# m_db_F     <- glex_obj_F$m
# m_db_F[, cols_rm] <- 0
# pred_db_F  <- rowSums(m_db_F) + glex_obj_F$intercept
# pred_org_F <- predict(xgb_class, X) 
# 
# MSE_F = (pred_db_F - as.numeric(task_freq$truth()))^2
# 
# mean(MSE_F)

#-------
# DATA    <- data_trans(data)
# set.seed(42)
# N       <- nrow(DATA)
# train_i <- sample(seq_len(N), size = 0.8 * N)
# 
# task_S_train    <- task_S$clone()$filter(train_i)
# task_S_test     <- task_S$clone()$filter(-train_i)
# task_freq_train <- task_freq$clone()$filter(train_i)
# task_freq_test  <- task_freq$clone()$filter(-train_i)
# 
# lrn_sev  <- best_at_s$clone(deep = TRUE)$train(task_S_train)
# lrn_freq <- best_at_f$clone(deep = TRUE)$train(task_freq_train)
# 
# pred_org_s <- lrn_sev$predict(task_S_test)$response
# pred_org_f <- lrn_freq$predict(task_freq_test)$prob[, 2]
# 
# X_test_s <- as.data.frame(task_S_test$data())
# y_test_s <- task_S_test$truth()
# X_test_f <- as.data.frame(task_freq_test$data())
# y_test_f <- task_freq_test$truth()
# 
# cols_rm <- c("Date_birth", "Date_driving_licence")
# m_s      <- glex_obj$m
# int_s    <- glex_obj$intercept
# m_s_db   <- m_s
# m_s_db[, cols_rm] <- 0
# pred_db_s <- rowSums(m_s_db) + int_s
# 
# m_f      <- glex_obj_F$m
# int_f    <- glex_obj_F$intercept
# m_f_db   <- m_f
# m_f_db[, cols_rm] <- 0
# pred_db_f <- rowSums(m_f_db) + int_f
# 
# y_test_s <- task_S_test$truth()
# y_test_f <- task_freq_test$truth()
# 
# mse_org_s <- mean((pred_org_s - y_test_s)^2)
# mse_db_s  <- mean((pred_db_s  - y_test_s)^2)
# mse_org_f <- mean((pred_org_f - y_test_f)^2)
# mse_db_f  <- mean((pred_db_f  - y_test_f)^2)
# 
# list(
#   severity  = c(original = mse_org_s, debiased = mse_db_s),
#   frequency = c(original = mse_org_f, debiased = mse_db_f)
# )
# 
# 
# 
