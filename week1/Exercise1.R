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
# 80/20
set.seed(1)

resampling <- mlr3::rsmp("holdout", ratio = 0.7)
resampling$instantiate(task)

train_idx <- resampling$train_set(1)
test_idx  <- resampling$test_set(1)



graph = po("encode")   %>>%
        po("scale")    %>>%
        lrn("classif.log_reg", predict_type = "prob")

learner <- GraphLearner$new(graph)

# her træner vi på træningsdata
learner$train(task, row_ids = train_idx)


prediction <- learner$predict(task, row_ids = test_idx)
res        <- prediction$score(msrs(c("classif.acc", "classif.auc")))

print(res)











