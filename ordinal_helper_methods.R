library(tidyverse)
library(ggplot2)
library(GGally)
library(glue)
library(plyr)
library(randomForest)
library(neuralnet)
library(caret)
library(gridExtra)
library(tcltk)

## Bin the data by distribution
bin_df <- function(df, bin_val, num_bins = 5){
  df %>% 
    mutate(class = cut(df[,bin_val],
                       breaks = c(quantile(df[,bin_val], probs = seq(0, 1, by = 1 / num_bins))), 
                       labels = FALSE, include.lowest = TRUE),
           bins = cut(df[,bin_val],
                      breaks = c(quantile(df[,bin_val], probs = seq(0, 1, by = 1 / num_bins))), 
                      include.lowest = TRUE)) -> df
  df$class <- as.factor(df$class)
  
  # cat(glue('Logger: Data binned by successfully by {bin_val}'), '\n')
  ## Assign average class median house value
  aggregate(formula(paste0(bin_val, "~class")), FUN = mean, data = df) %>%
    merge(x = df, y = ., by = "class") -> average_binned
  names(average_binned)[names(average_binned) == glue('{bin_val}.y')] <- "Class_Avg_Value"
  names(average_binned)[names(average_binned) == glue('{bin_val}.x')] <- bin_val
  
  return(average_binned)
}

train_model_grid <- function(data, method = "rf", binned_predictor_var = "Median_House_Value"){
  control <- trainControl(method = "repeatedcv",
                          number = 2, 
                          repeats = 1, 
                          search = "grid")
  if (method == "rf"){
    tunegrid <- expand.grid(.mtry=c(1:9))
    model <- train(as.numeric(class) ~ ., 
                   data = data,
                   method = method,
                   ntree = 25,
                   preProc = c("center", "scale"),
                   tuneGrid = tunegrid, 
                   trControl = control)
    cat('Model was trained successfully in 0 seconds', '\n')
    return(model)
  }
}

get_mse <- function(model){ 
  mean(model$residuals^2)
}

train_rf <- function(data, predict_column, mtry = 3){
  if(mtry > ncol(data) - 1){
    stop(glue('Invalid mtry. There are {ncol(data)-1} cols for training but mtry = {mtry}. Max mtry allowed: {ncol(data)-1}'))
  }
  else {
    start.time <- Sys.time()
    model <- randomForest(as.numeric(data[,predict_column]) ~ ., 
                   data = data,
                   mtry = mtry,
                   ntree = 25,
                   na.action = na.omit)
    end.time <- Sys.time()
    cat(glue('trained successfully in {round(end.time - start.time,2)} seconds'), '\n')
    return(model)
  }
}

get_all_models <- function(train, test, tune, binned_predictor_var, continuous_predictor_var, analysis_type, method = "rf"){
  pb <- txtProgressBar(min = 0,
                       max = length(tune),
                       style = 3,
                       width = 50,
                       char = "=")

  if (analysis_type == "post_pred"){
    all_preds_binned <- matrix(nrow = nrow(test), ncol = length(tune))
    all_preds_cont <- matrix(nrow = nrow(test), ncol = length(tune))
    for (i in 1:length(tune)){
      cat(glue('  Training Model {tune[i]} ... '))
      if (method == "rf"){
        model <- train_rf(train %>% select(-continuous_predictor_var), binned_predictor_var, tune[i])
      }
      else {
        stop("Error: Invalid Modeling Method (Acceptable Methods: rf")
      }
      bin_preds <- predict(model, newdata = test, type = "response")
      all_preds_binned[,i] <- bin_preds
      cont_lm <- lm(test[,continuous_predictor_var] ~ (as.numeric(test[,binned_predictor_var]) - bin_preds), test)
      if(length(predict(cont_lm, newdata = test)) == nrow(test)){
        all_preds_cont[,i] <- predict(cont_lm, newdata = test, type = "response")
      } else {
        cat('Logger: Model results in error')
        all_preds_cont[,i] <- rep(NA, nrow(test))
      }
      
      setTxtProgressBar(pb, i)
    }
    close(pb)
    results <- cbind(test[,binned_predictor_var], rowMeans(all_preds_binned),
                     test[,continuous_predictor_var], rowMeans(all_preds_cont)) %>% as.data.frame()
    colnames(results) <- c("Actual_Binned", "Pred_Binned", "Actual_Cont", "Pred_Cont")
  } else if (analysis_type == "post_hoc"){
    all_rs_binned <- c()
    all_mse_binned <- c()
    all_rs_cont <- c()
    all_mse_cont <- c()
    for (i in 1:length(tune)){
      cat(glue('  Training Model {tune[i]} ... '))
      if (method == "rf"){
        model <- train_rf(train %>% select(-continuous_predictor_var), binned_predictor_var, tune[i])
      }
      else {
        stop("Error: Invalid Modeling Method (Acceptable Methods: rf")
      }
      all_rs_binned[i] <- mean(model$rsq)
      all_mse_binned[i] <- mean(model$mse)
      
      cont_resids <- model$predicted
      cont_lm <- lm(train[,continuous_predictor_var] ~ cont_resids + as.factor(train[,binned_predictor_var]), train)
      all_rs_cont[i] <- summary(cont_lm)$r.squared
      all_mse_cont[i] <- get_mse(cont_lm)
  
      setTxtProgressBar(pb, i)
    }
    
    close(pb)
    results <- cbind(tune, all_rs_binned, all_mse_binned, all_rs_cont, all_mse_cont) %>% as.data.frame()
  } else {
    stop("Invalid Analysis Type")
  }
  return(results)
}


vis_results <- function(results_df, analysis_type){
  if (analysis_type == "post_pred"){
    g1 <- ggplot(results_df, aes(x = Actual_Binned, y = Pred_Binned)) + 
      geom_point() + 
      geom_smooth(se=F, method = "loess") +
      # geom_jitter() +
      theme_classic() +
      ggtitle(glue("Actual vs. Predicted Binned Labels"))
    g2 <- ggplot(results_df, aes(x = Actual_Cont, y = Pred_Cont)) + 
      geom_point() + 
      geom_smooth(se=F, method = "loess", col = "purple") +
      theme_classic() +
      ggtitle(glue("Actual vs. Predicted Binned Labels"))
    grid.arrange(g1, g2, ncol = 1)
  } else if (analysis_type == "post_hoc"){
    g1 <- ggplot(results_df, aes(x = tune, y = all_rs_binned)) + 
      geom_point() +
      geom_smooth(se=F, method = "loess") +
      theme_classic() +
      ggtitle(glue("Labeled R-Squared")) +
      xlab("mtry") +
      ylab("R-Squared")
    
    g2 <- ggplot(results_df, aes(x = tune, y = all_rs_cont)) + 
      geom_point() +
      geom_smooth(se=F, method = "loess", col = "purple") +
      theme_classic() +
      ggtitle(glue("Continuous R-Squared")) +
      xlab("mtry") +
      ylab("R-Squared")
    
    g3 <- ggplot(results_df, aes(x = tune, y = all_mse_binned)) + 
      geom_point() +
      geom_smooth(se=F, method = "loess") +
      theme_classic() +
      ggtitle(glue("Labeled MSE")) +
      xlab("mtry") +
      ylab("MSE")
    
    g4 <- ggplot(results_df, aes(x = tune, y = all_mse_cont)) + 
      geom_point() +
      geom_smooth(se=F, method = "loess", col = "purple") +
      theme_classic() +
      ggtitle(glue("Continuous MSE")) +
      xlab("mtry") +
      ylab("MSE (100k)")
    
    grid.arrange(g1, g2, g3, g4, ncol = 2)
  } else{
    stop("Invalid Analysis Type")
  }
}

## Main Method
run <- function(dataframe, features_for_training, continuous_predictor_var, 
                binned_predictor_var, tune, analysis_type = "post_hoc", method = "rf"){
  cat(glue('Logger: Entering Main Method for Training a {method} model'), '\n')
  cat('Logger: Binning response variable to create labels', '\n')
  binned_data <- bin_df(dataframe, continuous_predictor_var)
  binned_data$id <- 1:nrow(binned_data) %>% as.factor()
  
  cat(glue('Logger: Splitting data into training and test with split size 0.70'), '\n')
  train <- binned_data %>% sample_frac(0.70) %>% select(binned_predictor_var, features_for_training, id, continuous_predictor_var)
  test  <- anti_join(binned_data, train, by = 'id')
  train <- train %>% select(-id)
  
  cat(glue('Logger: Training model using method {method}'), '\n')
  cat(glue('Total Models to Train: {length(tune)}'), '\n')
  results <- get_all_models(train, test, tune, binned_predictor_var, continuous_predictor_var, analysis_type)
  
  cat('Logger: Visualizing model performance based on complexity')
  vis_results(results, analysis_type)
  return(results)
}




run_cv_sim <- function(folds = 10, cont_response = "CrimeRate", cont_location = 2){
  cat(glue('LOGGER: Entering Main Method for Training \n'))
  
  start_time = Sys.time()
  tree_set = c(1:10 %o% 10^(1:3))
  train_mat <- matrix(NA, nrow = folds, ncol = length(tree_set))
  test_mat <- matrix(NA, nrow = folds, ncol = length(tree_set))
  train_cont_mat <- matrix(NA, nrow = folds, ncol = length(tree_set))
  test_cont_mat <- matrix(NA, nrow = folds, ncol = length(tree_set))
  ind <- 1:(nrow(Train)/10)
  
  for (i in 1:folds){
    cat('\n')
    cat(glue('\n Training Model {i} \n'))
    train <- Train[ind,]
    test <- Train[-ind,]
    MDL <- gbm(class ~ ., data = train[, names(train) != cont_response], distribution = "gaussian", 
               n.trees = max(tree_set))
    for (j in 1:length(tree_set)){
      train_preds <- predict(MDL, newx = train[, (names(train) != cont_response) & 
                                                 (names(train) != "class")] %>% data.matrix(),
                             n.trees = tree_set[j])
      train_mat[i,j] <- MSE(y_pred = train_preds, y_true = as.numeric(train$class))
      
      test_preds <- predict(MDL, newx = test[, (names(test) != cont_response) & 
                                               (names(test) != "class")] %>% data.matrix(),
                            n.trees = tree_set[j])
      test_mat[i,j] <- MSE(y_pred = test_preds, y_true = as.numeric(test$class))
      
      binned_train_resids <- as.numeric(train$class) - train_preds
      binned_test_resids <- as.numeric(test$class) - test_preds
      
      cont_train_lm <- lm(train[,cont_location] ~ binned_train_resids + as.factor(train$class), data = train)
      cont_test_lm <- lm(test[,cont_location] ~ binned_test_resids + as.factor(test$class), data = test)
      
      train_cont_preds <- predict(cont_train_lm, type = "response")
      train_cont_mat[i,j] <- MSE(y_pred = train_cont_preds, y_true = as.numeric(train[,cont_response]))
      
      test_cont_preds <- predict(cont_test_lm, type = "response")
      test_cont_mat[i,j] <- MSE(y_pred = test_cont_preds, y_true = as.numeric(test[,cont_response]))
    }
    ind = ind + (nrow(Train)/10)
  }
  
  par(mfrow = c(2,2))
  plot(tree_set, colMeans(train_mat),
       col = "cornflowerblue",
       pch = 16,
       main = glue("CV Binned Train MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  plot(tree_set, colMeans(test_mat),
       col = "cornflowerblue",
       pch = 16,
       main = glue("CV Binned Test MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  # loess_values = loess(colMeans(train_cont_mat) ~ tree_set)
  plot(tree_set, colMeans(train_cont_mat),
       col = "deepskyblue",
       pch = 16,
       main = glue("CV Continuous Train MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  # lines(predict(loess_values), col = 'red', lwd = 2)
  # loess_values = loess(colMeans(test_cont_mat) ~ tree_set)
  plot(tree_set, colMeans(test_cont_mat),
       col = "deepskyblue",
       pch = 16,
       main = glue("CV Continuous Test MSE"),
       ylab = "MSE", 
       xlab = "Trees")
  # lines(predict(loess_values), col = 'red', lwd = 2)
  
  par(mfrow = c(2,2))
  end_time = Sys.time()
  cat('\n')
  cat(glue('LOGGER: Run Success in {round(end_time - start_time,2)}s \n'))
}