### Practical Machine Learning - quiz 4

## Question 1

# Load libraries and data sets
library(caret)
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 

# Set the variable y to be a factor variable in both the training and test set. Then set the seed to 33833.
vowel.train$y <- factor(vowel.train$y)
vowel.test$y <- factor(vowel.test$y)
set.seed(33833)

# Fit (1) a random forest predictor relating the factor variable y to the remaining variables and (2) a boosted predictor
# using the "gbm" method. Fit these both with the train() command in the caret package. 
model.rf <- train(factor(y) ~ ., data=vowel.train, method="rf")
model.gbm <- train(factor(y) ~ ., data=vowel.train, method="gbm")

# What are the accuracies for the two approaches on the test data set?
test.predictions.rf <- predict(model.rf, newdata=vowel.test)
test.predictions.gbm <- predict(model.gbm, newdata=vowel.test)
accuracy.rf <- sum(test.predictions.rf==vowel.test$y) / length(vowel.test$y)
accuracy.gbm <- sum(test.predictions.gbm==vowel.test$y) / length(vowel.test$y)
accuracy.rf
accuracy.gbm

# What is the accuracy among the test set samples where the two methods agree?
accuracy.mutual <- sum(test.predictions.rf == test.predictions.gbm) / length(vowel.test$y)
accuracy.mutual


## Question 2

# Load the Alzheimer's data
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# Set the seed to 62433 and predict diagnosis with all the other variables using a random forest ("rf"), boosted trees ("gbm")
# and linear discriminant analysis ("lda") model.
set.seed(62433)
model.1 <- train(factor(diagnosis) ~ ., data=training, method="rf")
model.2 <- train(factor(diagnosis) ~ ., data=training, method="gbm")
model.3 <- train(factor(diagnosis) ~ ., data=training, method="lda")

# Stack the predictions together using random forests ("rf").
predictions.1.training <- predict(model.1, newdata=training)
predictions.2.training <- predict(model.2, newdata=training)
predictions.3.training <- predict(model.3, newdata=training)
df.training.stacked <- data.frame(diagnosis=training$diagnosis, predictions.1=predictions.1.training, predictions.2=predictions.2.training, predictions.3=predictions.3.training)
model.stacked <- train(factor(diagnosis) ~ ., data=df.training.stacked, method="rf")
predictions.1.testing <- predict(model.1, newdata=testing)
predictions.2.testing <- predict(model.2, newdata=testing)
predictions.3.testing <- predict(model.3, newdata=testing)
df.testing.stacked <- data.frame(diagnosis=testing$diagnosis, predictions.1=predictions.1.testing, predictions.2=predictions.2.testing, predictions.3=predictions.3.testing)
predictions.stacked <- predict(model.stacked, df.testing.stacked)

# What is the resulting accuracy on the test set? Is it better or worse than each of the individual predictions?
accuracy.predictions.1.testing <- sum(predictions.1.testing==testing$diagnosis) / length(testing$diagnosis)
accuracy.predictions.2.testing <- sum(predictions.2.testing==testing$diagnosis) / length(testing$diagnosis)
accuracy.predictions.3.testing <- sum(predictions.3.testing==testing$diagnosis) / length(testing$diagnosis)
accuracy.predictions.stacked <- sum(predictions.stacked==testing$diagnosis) / length(testing$diagnosis)
accuracy.predictions.1.testing
accuracy.predictions.2.testing
accuracy.predictions.3.testing
accuracy.predictions.stacked


## Question 3

# Load the concrete data with the commands:
library(caret)
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

# Set the seed to 233 and fit a lasso model to predict Compressive Strength.
set.seed(233)
model.lasso.1 <- train(CompressiveStrength ~ ., data=training, method="lasso")

# Which variable is the last coefficient to be set to zero as the penalty increases? (Hint: it may be useful to look up ?plot.enet).
plot(model.lasso.1$finalModel)

# Alternative
predictors <- as.matrix(subset(training, select=-c(CompressiveStrength)))
response <- training$CompressiveStrength
model.lasso.2 <- enet(predictors, response, lambda=0)
plot.enet(model.lasso.2)


## Question 4

# Load the data on the number of visitors to the instructors blog
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv"
file <- "./gaData.csv"
download.file(url, file, method="curl")
library(lubridate)  # For year() function below
dat = read.csv("./gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

# Fit a model using the bats() function in the forecast package to the training time series.
library(forecast)
bats.model <- bats(tstrain)

# Then forecast this model for the remaining time points.
forecast <- forecast(bats.model, h=length(testing$visitsTumblr), level=95)
plot(forecast)

# For how many of the testing points is the true value within the 95% prediction interval bounds?
sum((forecast$lower <= testing$visitsTumblr) & (testing$visitsTumblr <= forecast$upper)) / length(testing$visitsTumblr)


## Question 5

# Load the concrete data with the commands:
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

# Set the seed to 325 and fit a support vector machine using the e1071 package to predict Compressive Strength
# using the default settings. 
library(e1071)
set.seed(325)
model.svm <- svm(CompressiveStrength ~ ., data=training)

# Predict on the testing set. What is the RMSE?
prediction.testing <- predict(model.svm, newdata=testing)
sqrt(sum((prediction.testing - testing$CompressiveStrength)^2) / length(testing$CompressiveStrength))
