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
df.training <- read.csv(file.training, na.strings=c("","#DIV/0!","NA"))
df.testing <- read.csv(file.testing, na.strings=c("","#DIV/0!","NA"))


## Exploring raw data sets

dim(df.training)
# Training set consists of 19622 measurements with 160 variables
dim(df.testing)
# Testing set consists of 20 measurements with 160 variables
names(df.training)

df.training$user_name
df.testing$user_name
# 6 persons have participated in experiments (variable: user_name), all 6 are present in training and test sets

df.training$classe
# Outcome variable that we want to predict is 'classe', factor variable with 5 different levels A, B, C, D, E

head(df.training, n=30)
# Training set contains time series measurements with time variables raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp
# and measured in time windows denoted by num_window. At the end of each window aggregate data is calculated
# (denoted by new_window=yes) and this is incorporated into the same dataset.

# Some plots based on training
qplot(num_window, pitch_belt, data=df.training, colour=classe, pch=user_name)
qplot(accel_forearm_x, accel_forearm_z, data=df.training, colour=classe, pch=user_name)


## Data Cleaning

# Remove variables from training dataset that are correlated to classe due to experiment setup
# Furthermore, only use non-aggregated data (new_window=="no")
df.training <- subset(df.training, new_window=="no", select=-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
df.testing <- subset(df.testing, new_window=="no", select=-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

# Remove columns with only NA values
df.training <- df.training[,colSums(is.na(df.training))<nrow(df.training)]
df.testing <- df.testing[,colSums(is.na(df.testing))<nrow(df.testing)]

# Verify class of variables in dataframe
lapply(df.training, class)
lapply(df.testing, class)


## Model Building: rpart

# Train model
model1 <- train(factor(classe) ~ ., method="rpart", data=df.training)

# Confusion table for training set
list.training.results1 <- predict(model1, newdata=df.training)
table(df.training$classe, list.training.results1)

# Calculate accuracy
sum(df.training$classe == list.training.results1) / length(list.training.results1)

# Check most important variables
varImp(model1)

# Plot a few important variables
qplot(pitch_forearm, roll_forearm, data=df.training, colour=classe)
qplot(magnet_dumbbell_y, magnet_dumbbell_z, data=df.training, colour=classe)

# Apply model to test set
list.testing.results1 <- predict(model1, newdata=df.testing)

# Plot results
df.testing.results1 <- df.testing; df.testing.results$classe=list.testing.results1
qplot(pitch_forearm, roll_forearm, data=df.testing.results1, colour=classe)



## Model Building: rf

# Train model
model2 <- train(factor(classe) ~ ., method="rf", data=df.training)

# Confusion table for training set
list.training.results2 <- predict(model2, newdata=df.training)
table(df.training$classe, list.training.results2)

# Calculate accuracy
sum(df.training$classe == list.training.results2) / length(list.training.results2)

# Check most important variables
varImp(model2)

# Plot a few important variables
qplot(pitch_forearm, roll_forearm, data=df.training, colour=classe)
qplot(magnet_dumbbell_y, magnet_dumbbell_z, data=df.training, colour=classe)

# Apply model to test set
list.testing.results2 <- predict(model2, newdata=df.testing)

# Plot results
df.testing.results2 <- df.testing; df.testing.results$classe=list.testing.results2
qplot(pitch_forearm, roll_forearm, data=df.testing.results2, colour=classe)