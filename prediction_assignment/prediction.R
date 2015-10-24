### Practical Machine Learning - Prediction assignment R code

## Preparations

# Variables
set.seed(12345)
dir.work <- "~/git/practical_machine_learning/prediction_assignment/data"
file.train <- "pml-training.csv"
file.predict <- "pml-testing.csv"
p.training <- 0.75

# Libraries
library(caret)
library(ggplot2)

# Load data
setwd(dir.work)
df.train.file <- read.csv(file.train, na.strings=c("","#DIV/0!","NA"))
df.predict.file <- read.csv(file.predict, na.strings=c("","#DIV/0!","NA"))

# Divide into training, testing and predicting set
m.train <- createDataPartition(df.train.file$classe, p=p.training, list = FALSE)
df.training <- df.train.file[m.train,]
df.testing <- df.train.file[-m.train,]
df.predicting <- df.predict.file


## Exploring raw data sets

dim(df.training)
# Training set consists of .. measurements with 160 variables
dim(df.testing)
# Validating set consists of .. measurements with 160 variables
names(df.training)

histogram(df.training$user_name)
histogram(df.testing$user_name)
# 6 persons have participated in experiments (variable: user_name), all 6 are present in training and test sets

histogram(df.training$classe)
# Outcome variable that we want to predict is 'classe', factor variable with 5 different levels A, B, C, D, E

head(df.training)
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


## Model 1: rpart

# Train model
model1 <- train(factor(classe) ~ ., method="rpart", data=df.training)

# Confusion table and accuracy for training set
list.training.results1 <- predict(model1, newdata=df.training)
table(df.training$classe, list.training.results1)
sum(df.training$classe == list.training.results1) / length(df.training$classe)

# Confusion table and accuracy for testing set
list.testing.results1 <- predict(model1, newdata=df.testing)
table(df.testing$classe, list.testing.results1)
sum(df.testing$classe == list.testing.results1) / length(df.testing$classe)

# Most important variables
varImp(model1)

# Save/load model to/from file
saveRDS(model1, "model_rpart.rds")
# model1 <- readRDS("model_rpart.rds")


## Model 2: gbm

# Train model
model2 <- train(factor(classe) ~ ., method="gbm", data=df.training)

# Confusion table and accuracy for training set
list.training.results2 <- predict(model2, newdata=df.training)
table(df.training$classe, list.training.results2)
sum(df.training$classe == list.training.results2) / length(df.training$classe)

# Confusion table and accuracy for testing set
list.testing.results2 <- predict(model2, newdata=df.testing)
table(df.testing$classe, list.testing.results2)
sum(df.testing$classe == list.testing.results2) / length(df.testing$classe)

# Most important variables
varImp(model2)

# Save/load model to/from file
saveRDS(model2, "model_gbm.rds")
# model2 <- readRDS("model_gbm.rds")


## Model 3: rf

# Train model
model3 <- train(factor(classe) ~ ., method="rf", data=df.training)

# Confusion table and accuracy for training set
list.training.results3 <- predict(model3, newdata=df.training)
table(df.training$classe, list.training.results3)
sum(df.training$classe == list.training.results3) / length(df.training$classe)

# Confusion table and accuracy for testing set
list.testing.results3 <- predict(model3, newdata=df.testing)
table(df.testing$classe, list.testing.results3)
sum(df.testing$classe == list.testing.results3) / length(df.testing$classe)

# Most important variables
varImp(model3)

# Save/load model to/from file
saveRDS(model3, "model_rf.rds")
# model3 <- readRDS("model_rf.rds")


## Making results more insightful

# Plot a few important variables
qplot(pitch_forearm, roll_forearm, data=df.training, colour=classe)
qplot(magnet_dumbbell_y, magnet_dumbbell_z, data=df.training, colour=classe)


## TODO: Choose best model (based on accurancy) and apply to predicting set
bestmodel <- model3
list.predicting.results <- predict(bestmodel, df.predicting)