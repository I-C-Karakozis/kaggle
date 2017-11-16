# Author: Yannis Karakozis
# 11/16/2017

library(glmnet)
setwd("C:/Users/Yannis/Documents/R/kaggle/titanic")

## setup for all regressions ##

prepare = function (data.orig) {
  data = data.orig[,c("Pclass","Sex", "Age","Embarked")]
  
  # one-hot encode
  encoded = model.matrix(~ (Sex-1) + (Embarked-1), data=data, 
                         contrasts.arg=list(Sex=contrasts(data$Sex, contrasts=F),
                                            Embarked=contrasts(data$Embarked, contrasts=F)))
  data = cbind(data.orig[,c("Pclass", "Age", "Fare")], encoded)
  
  # fill up null entries
  data[, "Age"][is.na(data[, "Age"])] <- mean(data[,"Age"], na.rm=TRUE)
  
  return(data)
}

# prepare design matrix and response vector
train.data.orig = read.csv("train.csv")
response = train.data.orig[,c("Survived")]
train.data = prepare(train.data.orig)
train.data = as.matrix(sapply(train.data, as.numeric))
train.data = train.data[,-8]
train.size = dim(train.data)[1]

## normalize and standardize data
norm_param = data.frame(offset = double(), scale = double()) 

# offset mean to zero
mu = mean(train.data[, "Age"], na.rm=TRUE)
train.data[, "Age"] = train.data[, "Age"] - mu
norm_param[1, 'offset'] = mu

mu = mean(train.data[, "Fare"], na.rm=TRUE)
train.data[, "Fare"] = train.data[, "Fare"] - mu
norm_param[2, 'offset'] = mu

# scale down by standard deviation
std = sd(train.data[, "Age"])
train.data[, "Age"] = train.data[, "Age"] / std
norm_param[1, 'scale'] = std

std = sd(train.data[, "Fare"])
train.data[, "Fare"] = train.data[, "Fare"] / std
norm_param[2, 'scale'] = std

# preprocess testing set
test.data.orig = read.csv("test.csv")
test.data = prepare(test.data.orig)
test.data = as.matrix(sapply(test.data, as.numeric))

# normalize and standardize data
test.data[, "Age"] = (test.data[, "Age"] - norm_param[1, 'offset']) / norm_param[1, 'scale']
test.data[, "Fare"] = (test.data[, "Fare"] - norm_param[2, 'offset']) / norm_param[2, 'scale']

#-----------------------------------------------------------#

## Linear Logistic Regression ##

cvfit<-cv.glmnet(train.data,response,family="binomial",type.measure="class", alpha=0)
lambda = cvfit$lambda.1se
lambda
# lambda:  0.1512091

# perform regularized logistic regression
fit = glmnet(train.data, response, family="binomial", lambda=lambda, alpha=0)
beta = fit$beta

# compute misclassification rate on training set
1 - sum(predict(fit, train.data, type="class") == response) / length(response)

# normalize and allocate predictions
Survived = predict(fit, test.data, type="class")
PassengerId = c((dim(train.data)[1]+1):(length(Survived)+dim(train.data)[1]))
df = data.frame(PassengerId, Survived)
write.csv(df, file = "predictions.csv", row.names=FALSE)

#-----------------------------------------------------------#