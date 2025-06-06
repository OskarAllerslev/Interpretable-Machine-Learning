

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
      Max_Date_birth         = max(Date_birth),
      Max_Date_driving_licence = max(Date_driving_licence),
      Max_Date_start_contract = max(Date_start_contract),
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
      Value_vehicle_mean     = mean(Value_vehicle),
      N_doors_mean           = mean(N_doors),
      Length_sum             = sum(Length, na.rm = TRUE)
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
data <- fread("~/Interpretable-Machine-Learning/aflevering/Motor vehicle insurance data.csv", sep = ";")
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
         min.node.size = 1,  # or your chosen stopping rule
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
         min.node.size = 5,  # or your chosen stopping rule
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
         max_depth     = 1,      
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
g_ranger_s_interpretable <- 
  prep_graph %>>%
  po("learner",
     lrn("regr.ranger",
         min.node.size = to_tune(10, 20),
         mtry          = to_tune(3, 5),
         num.trees     = to_tune(100, 200),
         importance    = "permutation",
         respect.unordered.factors = "order"
     )
  )
  
  

at_ranger_s_interp = AutoTuner$new(
  learner   = GraphLearner$new(g_ranger_s_interpretable, id = "ranger_interp"),
  resampling = rsmp("cv", folds = 2),
  measure    = msr("regr.mse"),                
  tuner      = tnr("random_search"),
  terminator = trm("evals", n_evals = 2),
  store_tuning_instance = TRUE
)

outer_rsmp_s = rsmp("cv", folds = 2)



kør_igen <- "nej"

if (!file.exists("~/Interpretable-Machine-Learning/aflevering/inte_pretable_severity.rds" ) | kør_igen == "ja") {
  rr_sev_interp = resample(
    task         = task_S,
    learner      = at_ranger_s_interp,
    resampling   = outer_rsmp_s,
    store_models = TRUE
  )
  saveRDS(rr_sev_interp, "~/Interpretable-Machine-Learning/aflevering/inte_pretable_severity.rds")
} else {
  rr_sev_interp = readRDS("~/Interpretable-Machine-Learning/aflevering/inte_pretable_severity.rds")
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
y_sev  <- task_S$truth()

## global feature-importance ----

expl_freq <- DALEX::explain(best_freq_mod, 
                            data = X_freq, 
                            y = y_freq, 
                            label = "XGB", 
                            weights = task_freq$weights$weight)
imp_freq <- DALEX::model_parts(expl_freq, type = "raw")
plot(imp_freq) 

### det her virker ikke rigtigt 


expl_sev  <- explain_mlr3(best_sev_mod,
                          data    = X_sev,
                          y       = y_sev,
                          label   = "Ranger",
                          weights = task_S$weights$weight)
imp_sev   <- model_parts(expl_sev, type = "raw")
plot(imp_sev) 






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


#### shaples for sev ----
# pfun_sev <- function(object, newdata) {
#   object$predict_newdata(newdata)$response  
# }
# 
# sh_sev <- fastshap::explain(
#   best_sev_mod,
#   X = as.data.frame(X_sev),
#   pred_wrapper = pfun_sev,
#   nsim = 500
# )





# 
# ## PDP og ICE  ---- Fungerer ikke endnu

# predictor_freq <- Predictor$new(
#   model = expl_freq$model,
#   data  = X_freq,
#   y     = y_freq,
#   predict.function = function(m, d) {
#     predict(expl_freq, newdata = d, type = "prob")[, 2]   # P(y = 1)
#   },
#   class = "classification"
# )
# 
# ## 2.2  PDP + ICE for én variabel (fx Cost_claims_sum_history)
# eff_freq <- FeatureEffect$new(
#   predictor = predictor_freq,
#   feature   = "Cost_claims_sum_history",   # brug præcis navn i X_freq
#   method    = "pdp+ice"
# )
# 
# p_freq <- plot(eff_freq) +
#   ggtitle("Frequency – PDP + ICE for Cost_claims_sum_history")
# 

## ale plots ----


######################################## frek ale 
best_par <- as.list(best_at_f$tuning_result$x)

final_xgb <- lrn(
  "classif.xgboost",
  predict_type = "prob",
  eval_metric  = "logloss",
  max_depth    = 1,                                   # fast fra din opskrift
  eta          = best_par[[1]]$classif.xgboost.eta,
  nrounds      = best_par[[1]]$classif.xgboost.nrounds,
  subsample    = best_par[[1]]$classif.xgboost.subsample
)

final_xgb$train(task_freq)





X_df <- task_freq$data(cols = final_xgb$feature_names) |> as.data.frame()

j <- which(names(X_df) == "Weight")  # præcis én kolonne



pred_freq <- function(X.model, newdata) {
  X.model$predict_newdata(newdata)$prob[, "1"]
}

ALEPlot(
  X        = X_df,
  X.model  = final_xgb,      # Learner
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

final_ranger <- lrn(
  "regr.ranger",
  predict_type   = "response",
  mtry           = best_par_s[[1]]$regr.ranger.mtry,
  min.node.size  = best_par_s[[1]]$regr.ranger.min.node.size,
  num.trees      = best_par_s[[1]]$regr.ranger.num.trees,
  importance     = "permutation"
)

final_ranger$train(task_S)               

pred_sev <- function(X.model, newdata) {
  X.model$predict_newdata(newdata)$response   
}

X_df  <- task_S$data(cols = final_ranger$feature_names) |> as.data.frame()
num_vars <- names(X_df)[sapply(X_df, is.numeric)]

ale_long <- map_dfr(
  num_vars,
  function(v) {
    j   <- which(names(X_df) == v)
    tmp <- tempfile(); png(tmp)          
    ale <- ALEPlot(X_df, final_ranger, pred_sev,
                   J = j, K = 40, NA.plot = TRUE)
    dev.off(); unlink(tmp)

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








