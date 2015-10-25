### Practical Machine Learning - Prediction assignment R code

## Preparations

# Variables
set.seed(1234)
dir.work <- "~/git/practical_machine_learning/prediction_assignment/data"
file.train <- "pml-training.csv"
file.predict <- "pml-testing.csv"
p.training <- 0.1

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

# Clear obsolete objects from memory
rm(df.predict.file, df.train.file)


## Exploring raw data sets

# dim(df.training)
# # Training set consists of .. measurements with 160 variables
# dim(df.testing)
# # Validating set consists of .. measurements with 160 variables
# names(df.training)
# 
# histogram(df.training$user_name)
# histogram(df.testing$user_name)
# # 6 persons have participated in experiments (variable: user_name), all 6 are present in training and test sets
# 
# histogram(df.training$classe)
# # Outcome variable that we want to predict is 'classe', factor variable with 5 different levels A, B, C, D, E
# 
# head(df.training)
# # Training set contains time series measurements with time variables raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp
# # and measured in time windows denoted by num_window. At the end of each window aggregate data is calculated
# # (denoted by new_window=yes) and this is incorporated into the same dataset.
# 
# # Some plots based on training
# qplot(num_window, pitch_belt, data=df.training, colour=classe, pch=user_name)
# qplot(accel_forearm_x, accel_forearm_z, data=df.training, colour=classe, pch=user_name)


## Data Cleaning

# Remove variables from training dataset that are correlated to classe due to experiment setup
# Furthermore, only use non-aggregated data (new_window=="no")
df.training <- subset(df.training, new_window=="no", select=-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
df.testing <- subset(df.testing, new_window=="no", select=-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

# Remove columns with only NA values
df.training <- df.training[,colSums(is.na(df.training))<nrow(df.training)]
df.testing <- df.testing[,colSums(is.na(df.testing))<nrow(df.testing)]

# # Verify class of variables in dataframe
# lapply(df.training, class)
# lapply(df.testing, class)


## Model 1: rpart

# rpart model gave poor results (accuracy=0.5), not used anymore

# # Train model
# model1 <- train(factor(classe) ~ ., method="rpart", data=df.training)
# 
# # Confusion table and accuracy for training set
# list.training.results1 <- predict(model1, newdata=df.training)
# table(df.training$classe, list.training.results1)
# sum(df.training$classe == list.training.results1) / length(df.training$classe)
# 
# # Confusion table and accuracy for testing set
# list.testing.results1 <- predict(model1, newdata=df.testing)
# table(df.testing$classe, list.testing.results1)
# sum(df.testing$classe == list.testing.results1) / length(df.testing$classe)
# 
# # Most important variables
# varImp(model1)
# 
# # Save/load model to/from file
# saveRDS(model1, "model_rpart.rds")
# # model1 <- readRDS("model_rpart.rds")


## Model 2: gbm

# gbm model gave memory allocation error, not used for now

# # Train model
# model2 <- train(factor(classe) ~ ., method="gbm", data=df.training)
# 
# # Confusion table and accuracy for training set
# list.training.results2 <- predict(model2, newdata=df.training)
# table(df.training$classe, list.training.results2)
# sum(df.training$classe == list.training.results2) / length(df.training$classe)
# 
# # Confusion table and accuracy for testing set
# list.testing.results2 <- predict(model2, newdata=df.testing)
# table(df.testing$classe, list.testing.results2)
# sum(df.testing$classe == list.testing.results2) / length(df.testing$classe)
# 
# # Most important variables
# varImp(model2)
# 
# # Save/load model to/from file
# saveRDS(model2, "model_gbm.rds")
# # model2 <- readRDS("model_gbm.rds")


## Model 3: rf

# Train model
#model.rf.1 <- train(factor(classe) ~ ., data=df.training, method="rf")
model.rf.2 <- train(factor(classe) ~ ., data=df.training, method="rf", prox = FALSE)
#model.rf.3 <- train(factor(classe) ~ ., data=df.training, method="rf", trControl=trainControl(method="cv", number=5), prox=TRUE, allowParallel=TRUE)


## Investigate model

# Set selected model and display
model.best <- model.rf.2
model.best
model.best$finalModel

# Confusion table and accuracy for training set
list.training.results <- predict(model.best, newdata=df.training)
table(df.training$classe, list.training.results)
sum(df.training$classe == list.training.results) / length(df.training$classe)

# Confusion table and accuracy for testing set
list.testing.results <- predict(model.best, newdata=df.testing)
table(df.testing$classe, list.testing.results)
sum(df.testing$classe == list.testing.results) / length(df.testing$classe)

# Most important variables
varImp(model.best)

# Save/load model to/from file
saveRDS(model.best, "model_rf.rds")
# model <- readRDS("model_rf.rds")


## Making results more insightful

# Plot a few important variables
qplot(roll_belt, pitch_forearm, data=df.training, colour=classe)
qplot(roll_belt, yaw_belt, data=df.training, colour=classe)
qplot(pitch_forearm, yaw_belt, data=df.training, colour=classe)


## Predict

# Apply best model to predicting set
list.predicting.results <- predict(model.best, df.predicting)
list.predicting.results

# Write to files
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(list.predicting.results)
