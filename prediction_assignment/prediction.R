### Practical Machine Learning - Prediction assignment R code

## Preparations

# Variables
set.seed(1234)
dir.work <- "~/git/practical_machine_learning/prediction_assignment/data"
file.train <- "pml-training.csv"
file.predict <- "pml-testing.csv"
# Set p small to speed up training process (with risk of lower accuracy)
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


## Exploring raw data sets

# Check data sets
dim(df.training)
dim(df.testing)
names(df.training)
head(df.training)
lapply(df.training, class)
lapply(df.testing, class)

# Some plots based on training
qplot(num_window, pitch_belt, data=df.training, colour=classe, pch=user_name)
qplot(accel_forearm_x, accel_forearm_z, data=df.training, colour=classe, pch=user_name)


## Data Cleaning

# Remove variables from training dataset that are correlated to classe due to experiment setup
# Furthermore, only use non-aggregated data (new_window=="no")
df.training <- subset(df.training, new_window=="no", select=-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
df.testing <- subset(df.testing, new_window=="no", select=-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

# Remove columns with only NA values
df.training <- df.training[,colSums(is.na(df.training))<nrow(df.training)]
df.testing <- df.testing[,colSums(is.na(df.testing))<nrow(df.testing)]


## Model 1: rpart

# rpart model gave poor results (accuracy=0.5), not used anymore
#model.rpart <- train(factor(classe) ~ ., method="rpart", data=df.training)


## Model 2: gbm

# gbm model gave memory allocation error, not used for now
#model.gbm <- train(factor(classe) ~ ., method="gbm", data=df.training)


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