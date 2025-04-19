library(mlr3)
library(mlr3learners )
library(mlr3tuning)
library(mlr3mbo)
library(glmnet)
library(OpenML)
library(mlr3pipelines)
library(future)
library(magrittr)
future::plan("multisession")

# exercise 1 default fitting ----

credit_data <- OpenML::getOMLDataSet(data.id = 31)
task <- mlr3::as_task_classif(credit_data$data, target = "class")

#### split data ------
set.seed(1)

graph = po("encode") %>>%
  po("scale")  %>>%
  lrn("classif.glmnet",
      predict_type = "prob",
      alpha  = to_tune(lower = 0, upper = 1),
      s      = to_tune(lower = 0, upper = 1))

glrn = GraphLearner$new(graph)
graph_learner_elastic_net = glrn
graph_learner_elastic_net$param_set$values = instance$result_learner_param_vals
graph_learner_elastic_net$param_set$values$classif.glmnet.lambda = graph_learner_elastic_net$param_set$values$classif.glmnet.s




# Brug classification error og 50 evalueringer
measure = mlr3::msr("classif.ce")
terminator = trm("evals", n_evals = 50)

# Tuning med random search
tuner = tnr("random_search")

instance = TuningInstanceBatchSingleCrit$new(
  task = task,
  learner = glrn,
  resampling = resampling,
  measure = measure,
  terminator = terminator
)

tuner$optimize(instance)

# Brug de bedste fundne parametre
graph_learner_elastic_net = glrn
graph_learner_elastic_net$param_set$values = instance$result_learner_param_vals
graph_learner_elastic_net$param_set$values$classif.glmnet.lambda = graph_learner_elastic_net$param_set$values$classif.glmnet.s

# Split i træning/test
set.seed(42)
split = partition(task, ratio = 0.8)
train_set = split$train
test_set  = split$test

# Træn og test
graph_learner_elastic_net$train(task, row_ids = train_set)
prediction = graph_learner_elastic_net$predict(task, row_ids = test_set)
test_error = prediction$score(msr("classif.ce"))

# Udskriv resultater
cv_error = instance$result_y

cat("Bedste alpha: ", round(as.numeric(instance$result_learner_param_vals$classif.glmnet.alpha), 3), "\n")
cat("Bedste s (lambda): ", round(as.numeric(instance$result_learner_param_vals$classif.glmnet.s), 3), "\n")
cat("CV Classification Error: ", round(cv_error, 3), "\n")
cat("Test Classification Error: ", round(test_error, 3), "\n")


# Print beta
graph_learner_elastic_net$model$classif.glmnet$model$beta


        

        
