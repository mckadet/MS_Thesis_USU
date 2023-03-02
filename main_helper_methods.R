
sample_randomly <- function(df, size = 1000){
  ind <- sample(1:nrow(df), size)
  return(list(df[ind,], df[-ind,]))
}

sample_by_distribution <- function(df, size = 1000){
  sub_size = size / unique(df$class) %>% length()
  one <- df[df$class == 1,]
  two <- df[df$class == 2,]
  three <- df[df$class == 3,]
  four <- df[df$class == 4,]
  five <- df[df$class == 5,]

  ind1 <- sample(1:nrow(one), sub_size)
  ind2 <- sample(1:nrow(two), sub_size)
  ind3 <- sample(1:nrow(three), sub_size)
  ind4 <- sample(1:nrow(four), sub_size)
  ind5 <- sample(1:nrow(five), sub_size)

  train <- rbind(
    one[ind1,],two[ind2,],three[ind3,],four[ind4,],five[ind5,]
  )
  test <- rbind(
    one[-ind1,],two[-ind2,],three[-ind3,],four[-ind4,],five[-ind5,]
  )
  return(list(train, test))
}


sample_by_distribution_ten <- function(df, size){
  sub_size = size / unique(df$class) %>% length()
  df_slice <- df[df$class == 1,]
  ind <- sample(1:nrow(df_slice), sub_size)
  train <- df_slice[ind,]
  test <- df_slice[-ind,]
    
  for (i in 2:length(unique(df$class))){
    df_slice <- df[df$class == i,]
    ind <- sample(1:nrow(df_slice), sub_size)
    train <- rbind(train, df_slice[ind,])
    test <- rbind(test, df_slice[-ind,])
  }
  return(list(train, test))
}


transform_preds <- function(test_preds, full, cont_response){
  (
    ( test_preds - min(test_preds) ) / ( max(test_preds) - min(test_preds) )
  ) * (max(full[,cont_response]) - min(full[,cont_response])) + min(full[,cont_response])
}

transform_ranks <- function(test_preds, df, nrows){
  v <- ceiling(nrows/5)
  df$preds <- test_preds
  transformed <- c()
  for(i in 1:max(as.numeric(df$class))){
    offset = v*i
    # cat(glue('Lower: {offset - v}, Upper: {offset}, Mid: {offset - (v/2)}'), '\n')
    subset <- df[df$class == i,]
    preds <- (subset$preds * v) - (v/2)
    transformed <- c(transformed, preds)
  }
  return(transformed)
}


sample_by_distribution_by_index <- function(df, ind, size = 1000){
  sub_size = size / unique(df$class) %>% length() 
  sample_one <- df[df$class == 1,] %>% sample_n(sub_size)
  sample_two <- df[df$class == 2,] %>% sample_n(sub_size)
  sample_three <- df[df$class == 3,] %>% sample_n(sub_size)
  sample_four <- df[df$class == 4,] %>% sample_n(sub_size)
  sample_five <- df[df$class == 5,] %>% sample_n(sub_size)
  return(rbind(sample_one, sample_two, sample_three, sample_four, sample_five))
}

add_noise <- function(df, variable, noise_factor = 0.5, plot = FALSE){
  new_df = df
  new_df$class <- jitter(as.numeric(df$class), noise_factor)
  if(plot){
    g1 <- ggplot(df, aes(as.numeric(class), df[,variable], col = as.factor(class))) + 
      geom_point() +
      theme_classic() +
      ggtitle("Response w/out Noise") +
      theme(legend.position = "none") +
      xlab("Continuous Response") +
      ylab("Class Labels")
    g2 <- ggplot(new_df, aes(as.numeric(class), df[,variable], col = as.factor(class))) + 
      geom_point() +
      theme_classic() +
      ggtitle("Response with Noise") +
      theme(legend.position = "none") + 
      xlab("Continuous Response") +
      ylab("Class Labels")
    grid.arrange(g1, g2, nrow = 2)
  }
  return(new_df)
}

RSQUARE = function(y_true,y_pred){
  cor(y_true, y_pred)^2
}



RF_performance <- function(train, test, cont_response, nrows){
  # test_ranks <- test$rank
  # train <- train[,names(train) != "rank"]
  # test <- test[,names(test) != "rank"]

  rf <- randomForest(
    as.numeric(class) ~ .,data = train[, names(train) != cont_response]
  )
  test_preds <- predict(rf, newdata = test[, !(names(test) %in% c("class",cont_response))])
  cont_test_lm <- lm(test[,cont_response] ~ test_preds, data = test)
  test_cont_preds <- predict(cont_test_lm, type = "response")
  return(sqrt(MSE(y_pred = test_cont_preds, y_true = test[,cont_response])))
  
  # cont_test_lm <- lm(test_ranks ~ test_preds, data = test)
  # test_cont_preds <- predict(cont_test_lm, type = "response")
  # return(sqrt(MSE(y_pred = test_cont_preds, y_true = test_ranks)))
}


RF_perc_change <- function(return_df, train, test, cont_response){
  best_binned_trees <- return_df[which.min(return_df$test_mat), "tree_set"] # 316
  cont_rmse <- return_df[return_df$tree_set == best_binned_trees, "test_cont_mat"]
  # RF_rmse <- RF_performance(train, test, cont_response)
  min_cont_rmse <- min(return_df$test_cont_mat)
  return(
      ((cont_rmse - min_cont_rmse) / min_cont_rmse)*100
    )
}


orchestrate_sim <- function(dataset, cont_response, sample_size, 
                            RMSE = TRUE, plot_title = "", num_classes = 5, 
                            MC = 25, mc_tree_length = 30, mc_min_trees = 1, 
                            mc_max_trees = 5, full = NA){

  full_start_time = Sys.time()
  
  binned_best_mat <- replicate(mc_tree_length, numeric(MC))
  cont_best_mat <- replicate(mc_tree_length, numeric(MC))
  mse_change_mat <- replicate(mc_tree_length, numeric(MC))
  rf_binned_test_mat <- vector(mode = "list", length = length(MC)) %>% unlist()
  
  for (i in 1:MC) {
    start_time = Sys.time()
    cat(glue('Begin Monte Carlo Iteration {i} of {MC}'), '\n')

    if (num_classes == 10){
      sub_sample <- sample_by_distribution_ten(dataset, size = sample_size)
    } else if (num_classes == 5){
      sub_sample <- sample_by_distribution(dataset, size = sample_size)
    } else{
      cat(glue("Number of classes {num_classes} is invalid."))
      return(NA)
    }
    # sub_sample <- sample_randomly(dataset, size = sample_size)
    train <- sub_sample[[1]]
    test <- sub_sample[[2]]
    return_df <- run_sim(train = sub_sample[[1]],
                         test = sub_sample[[2]],
                         cont_response = cont_response,
                         min_trees = mc_min_trees,
                         max_trees = mc_max_trees,
                         tree_length = mc_tree_length,
                         plot = FALSE, 
                         R2 = FALSE,
                         full = full)
    
    binned_best_mat[i,] <- return_df$test_mat
    cont_best_mat[i,] <- return_df$test_cont_mat
    mse_change_mat[i,] <- return_df$mse_perc_change
    rf_binned_test_mat[i] <- RF_performance(train, test, cont_response, nrows = nrow(dataset))
    
    cat(glue('--End Iteration {i} of {MC}: Completed in {round(Sys.time() - start_time,2)}'),'\n')
  }
  cat(glue('Full Run Success: Completed in {round(Sys.time() - full_start_time,2)}','\n'))
  cat('\n', glue('Baseline RF RMSE: {round(mean(rf_binned_test_mat),3)}'), '\n')
  cat(glue('Best Binned RMSE: {round(min(colMeans(binned_best_mat)),3)}'),  '\n')
  cat(glue('Best Cont RMSE: {round(min(colMeans(cont_best_mat)),3)}'), '\n')
  
  plot_return_mc(round(10^(seq(from=mc_min_trees, to=mc_max_trees, length=mc_tree_length))),
                 binned_best_mat, cont_best_mat, rf_binned_test_mat, RMSE = RMSE, title = plot_title)
}


orchestrate_sample_sim <- function(MC, SIZES, dataset, cont_response,
                                   MIN_TREES = 1, MAX_TREES = 3,TREE_LENGTH = 20, num_classes = 5){
  binned_best_trees_mat <- matrix(NA, nrow = MC, ncol = length(SIZES))
  cont_best_trees_mat <- matrix(NA, nrow = MC, ncol = length(SIZES))
  mse_perc_change_mat <- matrix(NA, nrow = MC, ncol = length(SIZES))
  # rf_mse_perc_change <- vector(mode = "list", length = length(MC)) %>% unlist()
  
  full_start_time = Sys.time()
  for(i in 1:MC){
    start_time = Sys.time()
    cat(glue('Begin Monte Carlo Iteration {i} of {MC}'), '\n')
    for (j in 1:length(SIZES)){
      cat(glue('----Begin Iteration {j} of {length(SIZES)} with n = {SIZES[j]}', '\n', '\n'))
      if (num_classes == 10){
        sub_sample <- sample_by_distribution_ten(dataset, size = SIZES[j])
      } else if (num_classes == 5){
        sub_sample <- sample_by_distribution(dataset, size = SIZES[j])
      } else{
        cat(glue("Number of classes {num_classes} is invalid."))
        return(NA)
      }
      # sub_sample <- sample_randomly(dataset, size = sample_size)
      train <- sub_sample[[1]]
      test <- sub_sample[[2]]
      return_df <- run_sim(train = train,
                           test = test,
                           cont_response = cont_response,
                           min_trees = MIN_TREES,
                           max_trees = MAX_TREES,
                           tree_length = TREE_LENGTH,
                           plot = FALSE)
      
      binned_thresh <- min(return_df$test_mat) * 1.01
      cont_thresh <- min(return_df$test_cont_mat) * 1.01
      binned_best_trees_mat[i,j] <- min(return_df[which(return_df$test_mat <= binned_thresh),
                                                  "tree_set"])
      cont_best_trees_mat[i,j] <- min(return_df[which(return_df$test_cont_mat <= cont_thresh),
                                                "tree_set"])
      # mse_perc_change_mat[i,j] <- mean(return_df$mse_perc_change)
      mse_perc_change_mat[i,j] <- RF_perc_change(return_df, train, test, cont_response)
    }
    # rf_mse_perc_change[i] <- RF_performance(size_sample[[1]], size_sample[[2]], cont_response = cont_response)
    cat(glue('End Iteration {i} of {MC}: Completed in {round(Sys.time() - start_time,2)}'),'\n')
  }
  cat(glue('Full Run Success: Completed in {round(Sys.time() - full_start_time,2)}'),'\n')
  return_plot <- plot_best_tree_mc(return_df, binned_best_trees_mat, cont_best_trees_mat, 
                                   mse_perc_change_mat, "Optimal Trees for Steel Data")
  
  mean_change <- round(mean_perc_change(binned_best_trees_mat, cont_best_trees_mat),3) * 100
  cat(glue('Mean Percentage Binned Complexity to Reach Continuous Optimal: {mean_change} %'),'\n')
  
  sd_change <- round(sd_perc_change(binned_best_trees_mat, cont_best_trees_mat),3) * 100
  cat(glue('SD Percentage Binned Complexity to Reach Continuous Optimal: {sd_change} %'),'\n')
  
  se_change <- round(SE_perc_change(binned_best_trees_mat, cont_best_trees_mat),3) * 100
  cat(glue('SE Percentage Binned Complexity to Reach Continuous Optimal: {se_change} %'),'\n')
  return(return_plot)
}


run_sim <- function(train, test, cont_response,
                    min_trees = 0, max_trees = 5, noise_added = FALSE, 
                    noise_factor = 0, tree_length = 15, plot=FALSE, R2 = FALSE, full = NA){
  tree_set = round(10^(seq(from=min_trees, to=max_trees, length=tree_length)))
  start_time = Sys.time()
  test_mat <- vector(mode = "list", length = length(tree_set)) %>% unlist()
  test_cont_mat <- vector(mode = "list", length = length(tree_set)) %>% unlist()
  pred_mat <- matrix(nrow = nrow(test), ncol = length(tree_set))
  # test_ranks <- test$rank
  # 
  # train <- train[,names(train) != "rank"]
  # test <- test[,names(test) != "rank"]

  MDL <- gbm(as.numeric(class) ~ ., data = train[, names(train) != cont_response], 
             distribution = "gaussian", n.trees = max(tree_set), n.cores = 8, 
             shrinkage = 0.3, interaction.depth = 1, keep.data = FALSE)

  for (i in 1:length(tree_set)){
    
    test_preds <- predict(MDL, newdata = test[, (names(test) != cont_response) & 
                                                (names(test) != "class")],
                          n.trees = tree_set[i])
    
    ## Cap off at 5.5 (or 10.5) and 0.5)
    test_preds[test_preds > max(as.numeric(test$class)) + 0.5 ] <- max(as.numeric(test$class)) + 0.5
    test_preds[test_preds < 0.5] <- 0.5
    
    if (R2){
      test_mat[i] <- RSQUARE(y_true = as.numeric(test$class), y_pred = test_preds)
    } else{
      test_mat[i] <- sqrt(MSE(y_pred = test_preds, y_true = as.numeric(test$class)))
    }
    # cat('\n')
    # cat(glue('R2 Test: {round(RSQUARE(y_true = as.numeric(test$class), y_pred = test_preds),3)}', '\n'))
    
    
    ## TRANSFORM BINNED PREDS TO CONT
    # binned_test_resids <- as.numeric(test$class) - test_preds
    # test_cont_preds <- transform_preds(test_preds, full, cont_response)
    
    ## RANKING
    # test_cont_preds <- transform_ranks(test_preds, test, nrows = nrow(full))
    # cont_test_lm <- lm(test_ranks ~ test_preds, data = test)
    
    ## USING LM
    # binned_test_resids <- as.numeric(test$class) - test_preds
    # cont_test_lm <- lm(test[,cont_response] ~ binned_test_resids + test$class, data = test)

    cont_test_lm <- lm(test[,cont_response] ~ test_preds, data = test)
    test_cont_preds <- predict(cont_test_lm, type = "response")
    
    
    # plot(test_ranks, test_cont_preds)
    # plot(as.numeric(test$class), test_preds)
    # cat(glue('Coeff: {cont_test_lm$coefficients[2]}'), '\n')
    # cat(glue('Correlation: {cor(test[,cont_response], test_cont_preds)}'), '\n')
    # cat(glue('Baseline: {sqrt(mean((test_cont_preds-mean(test_cont_preds))^2))}'), '\n')
    # plot(test[,cont_response], test_cont_preds, main = glue('Continuous w/ {tree_set[i]}'))
    # plot(as.numeric(test$class), test_preds, main = glue('Binned w/ {tree_set[i]}'))
    # plot(test_ranks, test_cont_preds)

    
    if (R2){
      test_cont_mat[i] <- RSQUARE(y_true = test[,cont_response], y_pred = test_cont_preds)
    } else{
      test_cont_mat[i] <- sqrt(MSE(y_pred = test_cont_preds, y_true = test[,cont_response]))
      # test_cont_mat[i] <- sqrt(MSE(y_pred = test_cont_preds, y_true = test_ranks))
    }

    if(plot){
      plot(x = test[,cont_response], 
           y = test_cont_preds,
           main = glue('Test vs Pred for {tree_set[i]} Trees'),
           xlab = glue("True {cont_response}"),
           ylab = glue("Predicted {cont_response}"))
      abline(coef=c(0,1))
    }
  }
  return_df <- data.frame(tree_set = tree_set,
             log10_tree_set = round(log10(tree_set),2),
             # train_mat = round(train_mat,2),
             test_mat = test_mat,
             # train_cont_mat = round(train_cont_mat,2),
             test_cont_mat = test_cont_mat
  )
  best_tree <- return_df[which.min(return_df$test_mat), "tree_set"] # 316
  cont_mse_from_binned <- return_df[return_df$tree_set == best_tree, "test_cont_mat"] #8.98
  best_cont_mse <- min(return_df$test_cont_mat)
  # best_cont_tree <- return_df[which.min(return_df$test_cont_mat), "tree_set"]
  
  
  return_df$mse_perc_change <- (
        (
          cont_mse_from_binned - best_cont_mse
        ) / best_cont_mse
  )* 100
  
  end_time = Sys.time()
  # cat('\n')
  # cat(glue('LOGGER: Run Success in {round(end_time - start_time,2)}s \n'))
  return(return_df)
}


plot_return_mc <- function(tree_set, best_bin, best_cont, rf_best, title = "", subtitle = "", RMSE = TRUE){
  cont_color = "cornflowerblue"
  if(RMSE){
    col_best_bin <- colMeans(best_bin)
    col_best_cont <- colMeans(best_cont)
    row_best_bin <- apply(best_bin, 1, FUN = min)
    row_best_cont <- apply(best_cont, 1, FUN = min)
    error = "Test RMSE"
  }
  else{
    col_best_bin <- colMeans(best_bin)
    col_best_cont <- colMeans(best_cont)
    error = "Test MSE"
  }
  comb <- as.data.frame(cbind(log10(tree_set), col_best_bin, col_best_cont))
  colnames(comb) <- c("trees", "best_bin", "best_cont")
  plot(log10(tree_set), col_best_bin,
       col = "grey",
       pch = 16, 
       cex = 0.3,
       main = glue("{title}"),
       sub = subtitle,
       ylab = error,
       xlab = "Log of # of Trees")
       # ylim = c(sqrt(mean(rf_best)) - .001, max(col_best_bin)+.001))
  legend("topright", legend=c("Binned", "Continuous", "RF"),
         col=c("grey", cont_color, "green"), lty=c(2,1,4), cex=0.9)
  lines(log10(tree_set), col_best_bin, lty=2, col = "grey")
  par(new=TRUE)
  plot(log10(tree_set), col_best_cont,
       col = cont_color,
       pch = 16,
       cex = 0.3,
       ylab = error,
       xlab = "Trees",
       axes = FALSE)
  lines(log10(tree_set), col_best_cont, lty=1, col = cont_color)
  
  min_cont <- min(col_best_cont)
  max_cont <- max(col_best_cont)
  tick <- (max_cont - min_cont) / 5
  ticks <- c(min_cont, 
             min_cont + tick,
             min_cont + tick*2,
             min_cont + tick*3,
             min_cont + tick*4,
             min_cont + tick*5)
  
  axis(4, at=round(ticks,2), col.axis="cornflowerblue", las=0)
  
  min_bin <- mean(tree_set[apply(best_bin, 1, FUN = which.min)])
  min_cont <- mean(tree_set[apply(best_cont, 1, FUN = which.min)])

  abline(v = log10(min_bin), col="grey", lty=5, lwd= 0.7)
  abline(v = log10(min_cont), col="blue", lty=5, lwd= 0.7)
  
  # abline(v = comb[which.min(col_best_bin), "trees"], col="grey", lty=5, lwd= 0.7)
  # abline(v = comb[which.min(col_best_cont), "trees"], col="cornflowerblue",lty=5, lwd = 0.7)
  abline(h = mean(rf_best), col = "green", lty = 4, lwd = 1)
  cat(glue('Avg_min_bin: {min_bin}'), '\n')
  cat(glue('Avg_min_cont: {min_cont}'), '\n')
}


plot_return_mc_R2 <- function(tree_set, best_bin, best_cont, title = "", subtitle = ""){
  cont_color = "cornflowerblue"

  col_best_bin <- colMeans(best_bin)
  col_best_cont <- colMeans(best_cont)
  plot(log10(tree_set), col_best_bin,
       col = "grey",
       pch = 16,
       main = glue("{title} (R2)"),
       sub = subtitle,
       ylab = "R Squared",
       xlab = "Trees")
  legend("bottomright", legend=c("Binned", "Cont"),
         col=c("grey", cont_color), pch=c(16,16), cex=0.9)
  lines(log10(tree_set), col_best_bin, pch=16, col = "grey")
  par(new=TRUE)
  plot(log10(tree_set), col_best_cont,
       col = cont_color,
       pch = 16,
       axes = FALSE)
  # axis(4, at = c(0,1), las = 0, col.axis="cornflowerblue")
  lines(log10(tree_set), col_best_cont, pch=16, col = cont_color)
  # min_cont <- min(col_best_cont)
  # max_cont <- max(col_best_cont)
  # tick <- (max_cont - min_cont) / 5
  # ticks <- c(min_cont, 
  #            min_cont + tick,
  #            min_cont + tick*2,
  #            min_cont + tick*3,
  #            min_cont + tick*4,
  #            min_cont + tick*5)
  axis(4, at=c(0.6, 0.7, 0.8, 0.9, 1), col.axis="cornflowerblue", las=0)
}



plot_best_tree_mc <- function(df, binned_best_trees_mat, cont_best_trees_mat, mse_perc_change_mat,
                              title = "",
                              noise_added = FALSE){
  # comb <- as.data.frame(
  #   cbind(log10(SIZES), log10(colMeans(binned_best_trees_mat)), 
  #         log10(colMeans(cont_best_trees_mat)), colMeans(mse_perc_change_mat))
  # )
  # colnames(comb) <- c("size", "Binned", "Cont", "MSE_Change")
  # comb %>% gather(key = "type", value = "value", -size) -> comb
  # ggplot(comb, aes(x = size, y = value)) + 
  #   geom_line(aes(color = type)) + 
  #   geom_point(aes(color = type)) +
  #   scale_color_manual(values = c("grey", "cornflowerblue", "#99FFFF")) +
  #   labs(title = title,
  #        x = "Log10 Sample Size",
  #        y = "Log10 Optimal Treees") +
  #   theme_classic()
  
  g1 <- ggplot(as.data.frame(log10(SIZES)), aes(x = log10(SIZES))) + 
    geom_line(aes(y = log10(colMeans(binned_best_trees_mat))), col = "grey") + 
    geom_point(aes(y = log10(colMeans(binned_best_trees_mat))), col = "grey") +
    geom_line(aes(y = log10(colMeans(cont_best_trees_mat))), col = "cornflowerblue") +
    geom_point(aes(y = log10(colMeans(cont_best_trees_mat))), col = "cornflowerblue") +
    # geom_histogram(aes(y = colMeans(mse_perc_change_mat)), 
    #                stat = "identity", alpha = .4, 
    #                fill = "lightgreen") +
    # scale_y_continuous(name = "log10 Optimal Trees", 
    #                    sec.axis = sec_axis(~.*5, name="MSE Change (%)")) +
    labs(title = "",
         x = "Log10 Sample Size",
         y = "Log10 Optimal Trees") +
    theme_classic() +
    theme(axis.title.y.right = element_text(color = "lightgreen"))
  g2 <- plot_mse_change(mse_perc_change_mat, SIZES)
  return_plot <- grid.arrange(g1, g2, nrow=2)
  return(return_plot)
}

plot_mse_change <- function(mse_perc_change_mat, SIZES){
  g <- ggplot(data = as.data.frame(colMeans(mse_perc_change_mat)),
              aes(x = log10(SIZES))) +
           geom_histogram(aes(y = colMeans(mse_perc_change_mat)),
                          stat = "identity",
                          alpha = .4,  fill = "lightgreen") + 
    theme_classic() +
    labs(x = "Log10 Sample Size",
         y = "RMSE Percent Change")
  return(g)
}


plot_features <- function(meltData, title = ""){
  ggplot(meltData, aes(factor(variable), value)) + 
    geom_boxplot(aes(fill = factor(variable)), ) + facet_wrap(~variable, scale="free", dir = "h") + 
    # theme(axis.text.x=element_blank()) +
    theme_classic() +
    labs(title = glue("Distribution of Quantitative Features for {title}"),
         x = "", y = "") +
    theme(aspect.ratio = 1,
          strip.text.x = element_blank(),
          axis.text.x = element_blank()) +
    scale_fill_brewer(palette = "Set1"
                      , name = "Variable")
}


mean_perc_change <- function(a, b){
  1 - (
    mean(
      (
        (
          colMeans(log(a)) - colMeans(log(b))
        ) / colMeans(log(a))
      )
    ) 
  )
}

sd_perc_change <- function(a, b){
    sd(
      (
        (
          colMeans(log(a)) - colMeans(log(b))
        ) / colMeans(log(a))
      ) 
    )
}

SE_perc_change <- function(a, b){
  (
    (
      sd(
        (
          (
            colMeans(log(a)) - colMeans(log(b))
          ) / colMeans(log(a))
        )
      )
    ) 
  ) / sqrt(length(colMeans(a)))
}


