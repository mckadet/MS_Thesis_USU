source('ordinal_helper_methods.R')
library(bannerCommenter)
library(tidyverse)

##################################################################
##                       Cal Housing Data                       ##
##################################################################

###################### Data Pre-Processing ####################### 
# Read in the data
df <- read.table('cal_housing_clean.csv', sep = ",", header = TRUE)
# Subset from 0 to 200k
df %>%
  filter(Median_House_Value >= 0 & Median_House_Value <= 200000) -> df
# Select features
features_for_training <- df %>%
  select(-Median_House_Value) %>%
  colnames() %>%
  as.vector()

######################## Model Training ##########################
run(dataframe = df,
    bin_var = "Median_House_Value",
    predictor_var = "class",
    tune = c(1:length(features_for_training)),
    features_for_training = features_for_training)




##################################################################
##                        Crime Data                            ##
##################################################################

###################### Data Pre-Processing ####################### 
# Read in the data
crime <- read.table('datasets/crime/crime.data', sep = ",", header = T)

# Take a random sample of predictors
features <- crime[,c(-1,-2,-3,-4)]
features$X1 <- as.factor(features$X1)
features_for_training <- sample(features, 20) %>% colnames() %>% as.vector()


binned <- bin_df(crime, bin_val = "X8")

boxplot(x = class, y = X8, data = binned)

############################## EDA ############################### 
hist(crime$X8, main = "Violent Crimes/100k Population", xlab = "Violent Crimes")

######################## Model Training ########################## 
results_df <- run(dataframe = crime, 
    continuous_predictor_var = "X8",
    binned_predictor_var = "class",
    tune = c(1:length(features_for_training)),
    features_for_training = features_for_training,
    analysis_type = "post_pred")
