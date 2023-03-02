## Binning Investigation
run_sim <- function(dataframe, folds = 10, cont_response = "CrimeRate", cont_location = 2){
  # cat(glue('LOGGER: Entering Main Method for Training \n'))
  tree_set = unique(round(10^(seq(from=0,to=4,length=40))))
  start_time = Sys.time()
  train_mat <- vector(mode = "list", length = length(tree_set))
  test_mat <- vector(mode = "list", length = length(tree_set))
  train_cont_mat <- vector(mode = "list", length = length(tree_set))
  test_cont_mat <- vector(mode = "list", length = length(tree_set))
  train_index <- sample(1:nrow(dataframe), nrow(dataframe)*0.7)
  
  train <- dataframe[-train_index,]
  test <- dataframe[train_index,]
  MDL <- gbm(class ~ ., data = train[, names(train) != cont_response], distribution = "gaussian", 
             n.trees = max(tree_set))
  for (i in 1:length(tree_set)){
    # cat('\n')
    # cat(glue('\n Training Model {i} \n'))
    train_preds <- predict(MDL, newx = train[, (names(train) != cont_response) & 
                                               (names(train) != "class")] %>% data.matrix(),
                           n.trees = tree_set[i])
    train_mat[i] <- MSE(y_pred = train_preds, y_true = as.numeric(train$class))
    
    test_preds <- predict(MDL, newx = test[, (names(test) != cont_response) & 
                                             (names(test) != "class")] %>% data.matrix(),
                          n.trees = tree_set[i])
    
    test_mat[i] <- MSE(y_pred = test_preds, y_true = as.numeric(test$class))
    
    
    
    binned_train_resids <- as.numeric(train$class) - train_preds
    binned_test_resids <- as.numeric(test$class) - test_preds
    
    cont_train_lm <- lm(train[,cont_location] ~ binned_train_resids + as.factor(train$class), data = train)
    cont_test_lm <- lm(test[,cont_location] ~ binned_test_resids + as.factor(test$class), data = test)
    
    train_cont_preds <- predict(cont_train_lm, type = "response")
    train_cont_mat[i] <- MSE(y_pred = train_cont_preds, y_true = as.numeric(train[,cont_response]))
    
    test_cont_preds <- predict(cont_test_lm, type = "response")
    test_cont_mat[i] <- MSE(y_pred = test_cont_preds, y_true = as.numeric(test[,cont_response]))
  }
  
  
  par(mfrow = c(2,2))
  plot(log(tree_set), train_mat,
       col = "cornflowerblue",
       pch = 16,
       main = glue("CV Binned Train MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  plot(log(tree_set), test_mat,
       col = "cornflowerblue",
       pch = 16,
       main = glue("CV Binned Test MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  plot(log(tree_set), train_cont_mat,
       col = "deepskyblue",
       pch = 16,
       main = glue("CV Continuous Train MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  plot(log(tree_set), test_cont_mat,
       col = "deepskyblue",
       pch = 16,
       main = glue("CV Continuous Test MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  
  par(mfrow = c(2,2))
  end_time = Sys.time()
  # cat('\n')
  # cat(glue('LOGGER: Run Success in {round(end_time - start_time,2)}s \n'))
}
# Read in the data
aba <- read.table('C:/Users/mckad/Documents/USU Coursework/Thesis Work/Ordinal Classification/datasets/abalone/abalone.data', sep = ",", header = T)
true_aba_names <- c("Sex", "Length", "Diameter", "Height", "Whole_wt", 
                    "Shucked_wt", "Viscera_wt", "Shell_wt", "Rings")
colnames(aba) <- true_aba_names

aba_binned <- bin_df(aba, "Rings")
aba_features <- aba_binned[,c(-11,-12)]
aba_encoded <- dummy_cols(aba_features, select_columns = 'Sex') %>% dplyr::select(-Sex)
run_sim(data = aba_encoded, cont_response = "Rings", cont_location = 10)




train_index <- sample(1:nrow(aba_encoded), nrow(aba_encoded)*0.7)
cont_response = "Rings"
train <- aba_encoded[train_index,]
test <- aba_encoded[-train_index,]
tree_set = unique(round(10^(seq(from=0,to=4,length=40))))
MDL <- gbm(class ~ ., data = train[, names(train) != cont_response], distribution = "gaussian", 
           n.trees = max(tree_set))

train_preds <- predict(MDL, newdata = train[, (names(train) != cont_response) & 
                                           (names(train) != "class")],
                       n.trees = max(tree_set))
MSE(y_pred = train_preds, y_true = as.numeric(train$class))

test_preds <- predict(MDL, newdata = test[, (names(test) != cont_response) & 
                                         (names(test) != "class")],
                      n.trees = max(tree_set))

MSE(y_pred = test_preds, y_true = as.numeric(test$class))



binned_train_resids <- as.numeric(train$class) - train_preds
binned_test_resids <- as.numeric(test$class) - test_preds
