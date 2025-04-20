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
# graph

graph = po("encode") %>>%
  po("scale") %>>%
  lrn("classif.glmnet",
      predict_type = "prob",
      alpha = to_tune(0,1),
      s = to_tune(0,1))

# create a graph learner

glrn = GraphLearner$new(graph)

# create autotuner

at = AutoTuner$new(
  learner = glrn,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  tuner = tnr("random_search"),
  terminator = trm("evals", n_evals = 50)
)

# above i have defined the first cross validation, and now we define the outer cross validation

outer_cv = rsmp("cv", folds = 5)
rr_nested = resample(task, at, outer_cv)
# above line resamples our task with respect to both the at and the outer cv


nested_ce = rr_nested$aggregate(msr("classif.ce"))
cat("nested cv classification error: ", base::round(nested_ce, 3), "\n")


