## Ordinal Regression using Cal.Housing

## Libraries Needed
library(tidyverse)
library(ggplot2)
library(GGally)
library(glue)
library(plyr)
library(randomForest)
library(neuralnet)

## Read in the data
df <- read.table("cal_housing.data", sep = ",")
colnames(df) <- c("Long", "Lat", "Housing_Median_Age", "Block_Rooms",
                  "Bedrooms", "Population", "Households", "Income", "Median_House_Value")

df$id <- 1:nrow(df) %>% 
  as.factor()


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
  
  ## Assign average class median house value
  aggregate(Median_House_Value ~ class, FUN = mean, data = df) %>%
    merge(x = df, y = ., by = "class") -> average_binned
    names(average_binned)[names(average_binned) == glue('{bin_val}.y')] <- "Class_Avg_Value"
    names(average_binned)[names(average_binned) == glue('{bin_val}.x')] <- bin_val
    
  return(average_binned)
}


binned_df <- bin_df(df, "Median_House_Value")
bins <- unique(binned_df$bins)
# (15k - 107k], (107k, 157k], (157k, 209k], (209k, 500k]


## EDA
hist(binned_df$Median_House_Value / 1000, 
     main = "Distribution of Median House Values",
     xlab = "House Value (in thousands)", breaks = 20)
abline(v = 15, lwd = 1, lty = 2, col = "blue")
abline(v = 107, lwd = 1, lty = 2, col = "blue")
abline(v = 157, lwd = 1, lty = 2, col = "blue")
abline(v = 209, lwd = 1, lty = 2, col = "blue")
abline(v = 290, lwd = 1, lty = 2, col = "blue")
abline(v = max(df$Median_House_Value)/1000, lwd = 1, lty = 2, col = "blue")

ggplot(binned_df, aes(x = as.factor(class), y = Median_House_Value, col = class)) + 
  geom_boxplot() + theme_classic()

ggplot(binned_df, aes(x=Long, y = Lat, col = Housing_Median_Age)) + 
  geom_point(alpha = 15, size = 2.5) + geom_tile() +
  scale_color_distiller(palette="Spectral",na.value=NA) +
  theme_bw() + ggtitle(glue("Spatial Dispersion"))

ggplot(binned_df, aes(x=Long, y = Lat, col = Block_Rooms)) + 
  geom_point(alpha = 15, size = 2.5) + geom_tile() +
  scale_color_distiller(palette="Spectral",na.value=NA) +
  theme_bw() + ggtitle(glue("Spatial Dispersion"))

ggplot(df, aes(x=Long, y = Lat, col = Median_House_Value)) + 
  geom_point(alpha = 15, size = 2.5) + geom_tile() +
  scale_color_distiller(palette="Spectral",na.value=NA) +
  theme_bw() + ggtitle(glue("Spatial Dispersion"))




###################### Continuous Regression #######################
# Linear Model
linear <- lm(class ~ Population + 
               Housing_Median_Age +
               Block_Rooms +
               Bedrooms +
               Households +
               Income, data = df)

summary(linear)

hist(linear$residuals)

## Predict
# preds <- predict(linear) / 1000
preds <- predict(linear)

## Map continuous preds back to labels
# Use mean
bins <- c(min(df$Median_House_Value)/1000, 107, 157, 209, 290, max(df$Median_House_Value)/1000)
pred_labels <- c()
for (i in 1:nrow(df)){
    pred_labels[i] <- case_when(
      preds[i] <= bins[2] ~ 1,
      preds[i] > bins[2] & preds[i] <= bins[3] ~ 2,
      preds[i] > bins[3] & preds[i] <= bins[4] ~ 3,
      preds[i] > bins[4] & preds[i] <= bins[5] ~ 4,
      preds[i] > bins[5] & preds[i] <= bins[6] ~ 5,
      TRUE ~ 6
    )
}


# Map linearly
linear_pred_labels <- preds / ((max(df$Median_House_Value) / 1000) / 5)


ggplot(df, aes(x = linear$residuals / 1000, y = Median_House_Value / 1000, col = as.factor(class))) + 
  geom_point() + 
  theme_classic() +
  xlab("Residual") + ylab("Median_House_Value")

ggplot(df, aes(x = as.factor(pred_labels), y = preds)) + 
  geom_boxplot() + 
  theme_classic() +
  xlab("Residual") + ylab("Median_House_Value")

table(df$class, df$Median_House_Value, )


Res <- -linear$residuals
LM <- lm(Median_House_Value~Res+as.factor(class), df)
Sum <- summary(LM)
Sum$r.squared

LM <- lm(Median_House_Value~as.factor(class), df)
Sum<-summary(LM)
Sum$r.squared



############### NEXT STEPS #############
# Binned y-values: 1-5. Try Avg. Group Value
# Partitions: by %, Equally spaced (limit values to 0 - 200k, then equally space groups)
# Best evaluation criteria: 
# Split into 70, 30 training and test
# Models to consider: ANN, RF, 
# Consider Adding Noise (only relevant with ANN)


############### Continous Regression Suite ################
get_mse <- function(y, y_preds){
  sum((y - y_preds)^2) / length(y)
}

reg_output <- function(model, binned_df){
  preds <- predict(model)
  res <- model$residuals
  resid <- binned_df$Median_House_Value - preds
  hist(resid)
  
  ggplot(binned_df, aes(x = resid / 1000, y = Median_House_Value / 1000, col = as.factor(class))) + 
    geom_point() + 
    theme_classic() +
    xlab("Residual") + ylab("Median_House_Value")
}

## Subset from 0 to 200k
df %>% 
  filter(Median_House_Value >= 0 & Median_House_Value <= 200000) -> subset

## Bin data by y, assign class and group average
new_bins <- bin_df(subset, "Median_House_Value")

## Split into train and tests
train <- new_bins %>% sample_frac(.70)
test  <- anti_join(new_bins, train, by = 'id')


## Predict on labels (1 - 5)
l1 <- lm(as.numeric(class) ~ Population + 
               Housing_Median_Age +
               Block_Rooms +
               Bedrooms +
               Households +
               Income, data = train)

rf1 <- randomForest(as.numeric(class) ~ Population + 
                     Housing_Median_Age +
                     Block_Rooms +
                     Bedrooms +
                     Households +
                     Income, data = train, mtry = 3, ntree = 100,
                   importance = TRUE, na.action = na.omit)

nn <- neuralnet(as.numeric(class) ~ Population + 
                  Housing_Median_Age +
                  Block_Rooms +
                  Bedrooms +
                  Households +
                  Income, data = train, hidden = 3,
                linear.output = FALSE)

y <- as.numeric(test$class)
l1_mse <- get_mse(y, predict(l1, newdata = test, type = "response"))
rf_mse <- get_mse(y, predict(rf1, newdata = test, type = "response"))
nn_mse <- get_mse(y, predict(nn, newdata = test, type = "response"))


## Predict on Average per class
l2 <- lm(Median_House_Value ~ Population + 
              Housing_Median_Age +
              Block_Rooms +
              Bedrooms +
              Households +
              Income, data = train)


rf2 <- randomForest(Median_House_Value ~ Population + 
                     Housing_Median_Age +
                     Block_Rooms +
                     Bedrooms +
                     Households +
                     Income, data = train, mtry = 3, ntree = 100)

nn2 <- neuralnet(Median_House_Value ~ Population + 
                  Housing_Median_Age +
                  Block_Rooms +
                  Bedrooms +
                  Households +
                  Income, data = train, hidden = 3,
                  linear.output = FALSE)

## Loop through nn using [2, 4, 8, 16, 32, 64] for hidden param
# save mse on predicting original discrete labels and on continous data


y <- test$Median_House_Value
l1_mse <- get_mse(y, predict(l2, newdata = test))
rf2_mse <- get_mse(y, predict(rf2, newdata = test))
nn2_mse <- get_mse(y, predict(nn2, newdata = test))




all_mse <- c(l1_mse, rf2_mse, nn2_mse)
complexity <- (c(1,2,3))
ggplot(as.data.frame(complexity), aes(x = complexity, y = all_mse)) + 
  geom_point() + 
  geom_smooth(se = F) +
  ggtitle("Model Performance by Complexity") +
  xlab("Model Complexity") +
  ylab("MSE") +
  theme_classic()

reg_output(l1, train)


# init_models <- function(response){
#   linear <- lm(response ~ Population + 
#              Housing_Median_Age +
#              Block_Rooms +
#              Bedrooms +
#              Households +
#              Income, data = train)
#   
#   rf <- randomForest(response ~ Population + 
#                        Housing_Median_Age +
#                        Block_Rooms +
#                        Bedrooms +
#                        Households +
#                        Income, data = train, mtry = 3, ntree = 100,
#                      importance = TRUE, na.action = na.omit)
#   
#   nn <- neuralnet(response ~ Population + 
#                     Housing_Median_Age +
#                     Block_Rooms +
#                     Bedrooms +
#                     Households +
#                     Income, data = train, hidden = 3,
#                   linear.output = FALSE)
#   return(list(linear, random, nn))
# }
# 
# init_models("Median_House_Value")


Res <- as.numeric(test$class) - predict(l1, newdata = test)
LM <- lm(Median_House_Value~Res+as.factor(class), test)
Sum <- summary(LM)
Sum$r.squared

LM2 <- lm(Median_House_Value ~ as.factor(class), test)
Sum<-summary(LM)
Sum$r.squared

sqrt(get_mse(test$Median_House_Value, predict(LM, newdata = test)))
sqrt(get_mse(test$Median_House_Value, predict(LM2, newdata = test)))



vertices <- c(2, 4, 8, 16, 32, 64)
all_discrete_mse <- c()
uninf_discrete_mse <- c()

all_cont_mse <- c()
for (i in length(vertices)){
  nn <- neuralnet(as.numeric(class) ~ Population + 
                     Housing_Median_Age +
                     Block_Rooms +
                     Bedrooms +
                     Households +
                     Income, data = train, hidden = vertices[i],
                   linear.output = FALSE)
  
  Res <- as.numeric(test$class) - predict(nn, newdata = test)
  LM <- lm(Median_House_Value~Res+as.factor(class), test)
  LM2 <- lm(Median_House_Value~as.factor(class), test)
  all_discrete_mse <- c(all_discrete_mse, get_mse(test$Median_House_Value, predict(LM, newdata = test)))
  uninf_discrete_mse <- c(all_discrete_mse, get_mse(test$Median_House_Value, predict(LM, newdata = test)))
  
}

nn2 <- neuralnet(Median_House_Value ~ Population + 
                   Housing_Median_Age +
                   Block_Rooms +
                   Bedrooms +
                   Households +
                   Income, data = train, hidden = 3,
                 linear.output = FALSE)
