### Practical Machine Learning - Prediction assignment R code

## Preparations

# Variables
dir.work <- "~/git/practical_machine_learning/prediction_assignment/data"
file.training <- "pml-training.csv"
file.testing <- "pml-testing.csv"

# Libraries
library(caret)
library(ggplot2)

# Load data
setwd(dir.work)
df.training.raw <- read.csv(file.training, na.strings=c("","#DIV/0!","NA"))
df.testing.raw <- read.csv(file.testing, na.strings=c("","#DIV/0!","NA"))


## Exploring raw data sets

dim(df.training.raw)
# Training set consists of 19622 measurements with 160 variables
dim(df.testing.raw)
# Testing set consists of 20 measurements with 160 variables
names(df.training.raw)

df.training.raw$user_name
df.testing.raw$user_name
# 6 persons have participated in experiments (variable: user_name), all 6 are present in training and test sets

df.training.raw$classe
# Outcome variable that we want to predict is 'classe', factor variable with 5 different levels A, B, C, D, E

head(df.training.raw, n=30)
# Training set contains time series measurements with time variables raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp
# and measured in time windows denoted by num_window. At the end of each window aggregate data is calculated
# (denoted by new_window=yes) and this is incorporated into the same dataset.

# Some plots based on training.raw
qplot(num_window, pitch_belt, data=df.training.raw, colour=classe, pch=user_name)
qplot(accel_forearm_x, accel_forearm_z, data=df.training.raw, colour=classe, pch=user_name)


## Data Cleaning

# Remove variables from training dataset that are correlated to classe due to experiment setup
# Furthermore, only use non-aggregated data (new_window=="no")
df.training.filtered <- subset(df.training.raw, new_window=="no", select=-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
df.testing.filtered <- subset(df.testing.raw, new_window=="no", select=-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

# Remove columns with only NA values
df.training.clean <- df.training.filtered[,colSums(is.na(df.training.filtered))<nrow(df.training.filtered)]
df.testing.clean <- df.testing.filtered[,colSums(is.na(df.testing.filtered))<nrow(df.testing.filtered)]

# Verify class of variables in dataframe
lapply(df.training.clean, class)
lapply(df.testing.clean, class)


## Model Building

# Apply tree model
model.rpart <- train(factor(classe) ~ ., method="rpart", data=df.training.clean)

# Confusion table for training set
list.training.results <- predict(model.rpart, newdata=df.training.clean)
table(df.training.clean$classe, list.training.results)

# Calculate accuracy
sum(df.training.clean$classe == list.training.results) / length(list.training.results)

# Check most important variables
varImp(model.rpart)

# Plot a few important variables
qplot(pitch_forearm, roll_forearm, data=df.training.clean, colour=classe)
qplot(magnet_dumbbell_y, magnet_dumbbell_z, data=df.training.clean, colour=classe)


## Apply model to test set
list.testing.results <- predict(model.rpart, newdata=df.testing.clean)
df.testing.withresults <- df.testing.clean; df.testing.withresults$classe=list.testing.results
qplot(pitch_forearm, roll_forearm, data=df.testing.clean)
