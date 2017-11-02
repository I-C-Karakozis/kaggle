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
  encoded = encoded[, -3]
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
# with Lambdas (0.01, 0.01, 1, 10, 100, 1000)
# to determine order of magnitude
Lambdas = c(1, 10, 25, 50, 75, 100)
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
lambda = Lambdas[which.min(CV)]
# lambda = 10

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
rss_unscaled = rss * (norm_param[1, 'scale'])^2

rss
rss / dim(train.data)[1]
rss_unscaled
rss_unscaled / dim(train.data)[1]
# rss (scaled): 796.7965
# mean rss (scaled): 0.5311977
# rss_unscaled: 9.46513e+13
# mean rss_unscaled: 6.310087e+10

# compute RSS on testing set
rss = 0
for (s in 1:length(test.data.price)){
  pred = 0
  for (m in 1:length(split.price)) {
    pred = pred + alpha[m] * kernel(test.data[s, ], split.train[m, ])
  }
  rss = rss + (pred - test.data.price[s]) ^ 2
}
rss_unscaled = rss * (norm_param[1, 'scale'])^2

rss
rss / dim(test.data)[1]
rss_unscaled
rss_unscaled / dim(test.data)[1]
# rss (scaled): 3640.563
# mean rss (scaled): 0.5614687
# rss_unscaled: 4.324618e+14
# mean rss_unscaled: 6.669676e+10

# predict price of gates house
pred = 0
for (m in 1:length(split.price)) {
  pred = pred + alpha[m] * kernel(gates.data, split.train[m, ])
}
pred = pred * norm_param[1, 'scale'] + norm_param[1, 'offset']
# prediction: $14,667,813 (similar to OLS; undervalued)

#-----------------------------------------------------------#