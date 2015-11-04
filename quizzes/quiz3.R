### Practical Machine Learning - quiz 3
### https://class.coursera.org/predmachlearn-033/quiz/attempt?quiz_id=93


## Question 1

# Preparations
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

# 1. Subset the data to a training set and testing set based on the Case variable in the data set. 
#training <- segmentationOriginal[segmentationOriginal$Case=='Train',]
#testing <- segmentationOriginal[segmentationOriginal$Case=='Test',]
training <- subset(segmentationOriginal, Case=='Train', select=-c(Case))
testing <- subset(segmentationOriginal, Case=='Test', select=-c(Case))

# 2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings. 
set.seed(125)
modelFit <- train(Class ~ ., method="rpart", data=training)

# 3. In the final model what would be the final model prediction for cases with the following variable values:
#    a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2 
#    b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100 
#    c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100 
#    d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2 
library(rattle)
fancyRpartPlot(modelFit$finalModel)
modelFit$finalModel
# a -> PS
# b -> WS
# c -> PS
# d -> <NA>


## Question 3

# Preparations
library(caret)
library(pgmm)
# These data contain information on 572 different Italian olive oils from multiple regions in Italy.
data(olive)
olive = olive[,-1]

# Fit a classification tree where Area is the outcome variable.
modelFit <- train(Area ~ ., method="rpart", data=olive)

# Then predict the value of area for the following data frame using the tree command with all defaults
#    newdata=as.data.frame(t(colMeans(olive)))
# What is the resulting prediction? Is the resulting prediction strange? Why or why not?
predict(modelFit, newdata=as.data.frame(t(colMeans(olive))))

# Better result
modelFit2 <- train(factor(Area) ~ ., method="rpart", data=olive)
predict(modelFit2, newdata=as.data.frame(t(colMeans(olive))))


## Question 4

# Load the South Africa Heart Disease Data and create training and test sets with the following code:
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

# Then set the seed to 13234 and fit a logistic regression model (method="glm", be sure to specify family="binomial")
# with Coronary Heart Disease (chd) as the outcome and age at onset, current alcohol consumption,
# obesity levels, cumulative tabacco, type-A behavior, and low density lipoprotein cholesterol as predictors.
set.seed(13234)
modelSAheartglm <- train(factor(chd) ~ age + alcohol + obesity + tobacco + typea + ldl, method="glm", family="binomial", data=trainSA)
summary(modelSAheartglm)
table(trainSA$chd, predict(modelSAheartglm, newdata=trainSA))

# Calculate the misclassification rate for your model using this function and a prediction on the "response" scale:
missClass = function(values,prediction){sum(prediction != values)/length(values)}
# What is the misclassification rate on the training set?
missClass(trainSA$chd, predict(modelSAheartglm, newdata=trainSA))
# What is the misclassification rate on the test set?
missClass(testSA$chd, predict(modelSAheartglm, newdata=testSA))


## Question 5

# Load the vowel.train and vowel.test data sets
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 

# Set the variable y to be a factor variable in both the training and test set.
vowel.train$y <- factor(vowel.train$y)
vowel.test$y <- factor(vowel.test$y)

# Then set the seed to 33833.
set.seed(33833)

# Fit a random forest predictor relating the factor variable y to the remaining variables.
modelFitvowel <- train(factor(y) ~ ., method="rf", data=vowel.train)

# Read about variable importance in random forests here: http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
# The caret package uses by defualt the Gini importance.
# Calculate the variable importance using the varImp function in the caret package.
# What is the order of variable importance?
varImp(modelFitvowel)

vowel.rf <- randomForest(factor(y) ~ ., data=vowel.train, ntree=1000, keep.forest=FALSE, importance=TRUE)
plot(vowel.rf, log="y")
varImpPlot(vowel.rf)