# Author: Yannis Karakozis
# ORF 350 HW4, Problem 3
# 10/21/2017

## setup for all regressions ##

prepare = function (data.orig) {
  data = data.orig[,c("Pclass","Sex", "Age","SibSp", "Parch", "Fare", "Embarked")]
  
  # one-hot encode
  encoded = model.matrix(~ (Sex-1) + (Embarked-1), data=data, 
                         contrasts.arg=list(Sex=contrasts(data$Sex, contrasts=F),
                                            Embarked=contrasts(data$Embarked, contrasts=F)))
  data = cbind(data.orig[,c("Pclass", "Age","SibSp", "Parch", "Fare")], encoded)
  
  # fill up null entries
  data[, "Age"][is.na(data[, "Age"])] <- mean(data[,"Age"], na.rm=TRUE)
  
  return(data)
}

set.seed(10)

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

# prepare for cross validation
J = 5
split.size = train.size / J

# preprocess testing set
test.data.orig = read.csv("test.csv")
test.data = prepare(test.data.orig)
test.data = as.matrix(sapply(test.data, as.numeric))

# normalize and standardize data
test.data[, "Age"] = (test.data[, "Age"] - norm_param[1, 'offset']) / norm_param[1, 'scale']
test.data[, "Fare"] = (test.data[, "Fare"] - norm_param[2, 'offset']) / norm_param[2, 'scale']

#-----------------------------------------------------------#

## Linear Kernel Regression ##

# pick set of lambdas
# range identified after performing cross-validation
# with Lambdas (0.001, 0.01, 1, 10, 100, 1000)
# to determine order of magnitude
Lambdas = c(0.001, 0.01, 1, 10, 100, 1000)
k = length(Lambdas)

# create kernel and kernel matrix
kernel = function (u, v) {
  as.numeric(u %*% v)
}

# perform cross validation
CV = numeric(k)
for (i in 1:k) {
  for (j in 1:J) {
    # prepare training set
    split.train = train.data[-(((j-1)*split.size + 1):(j * split.size)),]
    split.price = response[-(((j-1)*split.size + 1):(j * split.size))]
    
    # prepare validation set
    split.valid = train.data[((j-1)*split.size + 1):(j * split.size), ]
    split.price.valid = response[((j-1)*split.size + 1):(j * split.size)]
    
    # compute kernel matrix
    K = matrix(0, nrow=length(split.price), ncol=length(split.price))
    for (row in 1:dim(K)[1]) {
      for (col in 1:dim(K)[2]) {
        K[row,col] = kernel(split.train[row,], split.train[col,])
      }
    }
    
    # compute dual variables vector
    alpha = solve(K + Lambdas[i] * diag(dim(K)[1])) %*% split.price 
    
    # compute score
    ds = 0
    for (s in 1:length(split.price.valid)){
      pred = 0
      for (m in 1:length(split.price)) {
        pred = pred + alpha[m] * kernel(split.valid[s, ], split.train[m, ])
      }
      ds = ds + (pred - split.price.valid[s]) ^ 2
    }
    
    # update CV(k)
    CV[i] = CV[i] + ds / length(split.price.valid)
  }
}

# find best performing lambda
CV = CV / J
lambda = 0.05#Lambdas[which.min(CV)]
lambda


# fit the model using the entire training set
split.train = train.data
split.price = response

# compute kernel matrix
K = matrix(0, nrow=length(split.price), ncol=length(split.price))
for (row in 1:dim(K)[1]) {
  for (col in 1:dim(K)[2]) {
    K[row,col] = kernel(split.train[row,], split.train[col,])
  }
}

# compute dual variables vector
alpha = solve(K + lambda * diag(dim(K)[1])) %*% split.price 

# compute RSS on training set
rss = 0
for (s in 1:length(split.price)){
  pred = 0
  for (m in 1:length(split.price)) {
    pred = pred + alpha[m] * kernel(split.train[s, ], split.train[m, ])
  }
  rss = rss + (pred - split.price[s]) ^ 2
}
# rss_unscaled = rss * (norm_param[1, 'scale'])^2

rss
rss / dim(train.data)[1]
# rss (scaled): 126.7726
# mean rss (scaled): 0.1422812

# predictions on testing set
predictions = numeric(dim(test.data)[1])
for (s in 1:dim(test.data)[1]){
  pred = 0
  for (m in 1:length(split.price)) {
    pred = pred + alpha[m] * kernel(test.data[s,], split.train[m, ])
  }
  predictions[s] = pred
}

# normalize and allocate predictions
predictions[][is.na(predictions[])] <- min(predictions, na.rm=TRUE)
Survived = round(predictions)
PassengerId = c((dim(train.data)[1]+1):(length(Survived)+dim(train.data)[1]))
df = data.frame(PassengerId, Survived)
write.csv(df, file = "predictions.csv", row.names=FALSE)

#-----------------------------------------------------------#