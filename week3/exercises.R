library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(OpenML)
library(mlr3pipelines)
library(future)
future::plan("multisession") 
library(tidyverse)
# load credit-g data and define task

credit_data = getOMLDataSet(data.id = 31)


task = as_task_classif(credit_data$data, target = "class") 

#### split data ------
# 80/20
set.seed(1)

resampling <- mlr3::rsmp("holdout", ratio = 0.7)
resampling$instantiate(task)

train_idx <- resampling$train_set(1)
test_idx  <- resampling$test_set(1)

graph = po("encode")   %>>%
  po("scale")    %>>%
  #lrn("classif.rpart", predict_type = "prob")
  lrn("classif.rpart", xval = 5, predict_type = "prob")

learner <- GraphLearner$new(graph)

# her træner vi på træningsdata
learner$train(task, row_ids = train_idx)


prediction <- learner$predict(task, row_ids = test_idx)
res        <- prediction$score(msrs(c("classif.acc", "classif.auc")))

print(res)


# load credit-g data and define task
full_tree_trained <- learner$model$classif.rpart$model
plot(full_tree_trained , compress = TRUE, margin = 0.1)
text(full_tree_trained , use.n = TRUE, cex = 0.8)

# load credit-g data and define task
rpart::plotcp(full_tree_trained)
rpart::printcp(full_tree_trained)

best_cp <- 0.031  # for example, after inspecting the table
pruned_learner <- lrn("classif.rpart", cp = best_cp)
pruned_learner$train(task)

plot(pruned_learner$model, compress = TRUE, margin = 0.1)
text(pruned_learner$model, use.n = TRUE, cex = 0.8)

#________________________________-- Bencmark

learners <- list(
  
  # A baseline model that predicts the majority class
  lrn("classif.featureless", id = "featureless"),
  
  # A CART decision tree (non-pruned)
  lrn("classif.rpart", id = "rpart_unpruned", predict_type = "prob"),
  
  # A pruned decision tree with best_cp
  lrn("classif.rpart", cp = best_cp, id = "rpart_pruned", predict_type = "prob"),
  
  # Auto-tuned xgboost
  auto_tuner(
    lrn("classif.xgboost", predict_type = "prob"),
    resampling = rsmp("cv", folds = 3),
    measure = msr("classif.ce"),
    search_space = ps(
      eta = p_dbl(0, 0.5),
      nrounds = p_int(10, 5000),
      max_depth = p_int(1, 10)
    ),
    terminator = trm("evals", n_evals = 20),
    tuner = tnr("random_search"),
    id = "xgboost_tuned"
  ),
  
  # Auto-tuned random forest
  auto_tuner(
    lrn("classif.ranger", predict_type = "prob"),
    resampling = rsmp("cv", folds = 3),
    measure = msr("classif.ce"),
    search_space = ps(
      mtry.ratio = p_dbl(0.1, 1),
      min.node.size = p_int(1, 50)
    ),
    terminator = trm("evals", n_evals = 20),
    tuner = tnr("random_search"),
    id = "ranger_tuned"
  )
)

# Define the resampling strategy
resampling <- rsmp("cv", folds = 5)

# Create the benchmark grid
design <- benchmark_grid(
  tasks = task,
  learners = learners,
  resamplings = resampling
)

# Run the benchmark
res <- benchmark(design)



res$aggregate(list(
  msr("classif.ce"),    # Classification error
  msr("classif.acc"),   # Accuracy
  msr("classif.auc"),   # AUC
  msr("classif.fpr"),   # False Positive Rate
  msr("classif.fnr")    # False Negative Rate
))


