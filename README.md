# Summer Institute in Statistics for Big Data Workshop: Supervised Machine Learning
## Seattle, WA - July 18-20, 2018
Notes and R code

### Wednesday July 18, 2018 1:30p - 5:00p
- y = outcomes, x = design matrix or covariates, n = number of observations, p = number of covariates
- betas = for every 1 unit increase in X, Y changes by beta assuming all other variables remain constant
- low dimensional - few covariates, lots of observations, n>>p
- high dimensional - era of big data - n = p, or n<<p
- can covariates be combined - discrete (and/or ordered) and continuous data
- too many covariates results in overfitting
- when p = n, or p > n, overfitting is almost guanteed
    - overfitting results from degrees of freedom...
- more difficult to design how many samples are needed, no p-values, or concepts of power
- unsupervised learning occurs for clustering or dimensionality reduction, Y is unknown

```R
# The R 'outer', 'contour', and 'image' function
x = seq(-pi, pi, length = 50)
y = x

# Makes a new number from each combiantion of X and Y
f = outer(x, y, function(x, y)cos(y) / (1 + x^2))

# Contour plot - like a topo map
contour(x, y, f)
contour(x, y, f, nlevels = 45, add = T)
fa = (f - t(f)) / 2
contour(x, y, fa, nlevels = 15)

# Image produce a plot similar to a heatmap
image(x, y, fa)

# 3D plot of same data
persp(x, y, fa)
persp(x, y, fa, theta = 30)
persp(x, y, fa, theta = 30, phi = 20)
persp(x, y, fa, theta = 30, phi = 70)
persp(x, y, fa, theta = 30, phi = 40)

# 'identifiy'
identify(horsepower, mpg, name)
# Allows you to click on points on a plot to identify the particular point.  Use CNTRL+click to quit.

# 'fix' command allows one to make small changes in a table!
fix(data)
```

Regression vs classification
- Regression - predict a quantitative or continuous response
- classification - predict a categorical response
- the Beta coefficients must be linear!  We can transform covariates as needed
- classical way to fit a model is to minimize the sum (across samples, n) of the least squares

Least squares in R Lab
```R
library(MASS)
library(ISLR)

# Simple Linear Regression
fix(Boston)
names(Boston)
lm.fit = lm(medv ~ lstat, data = Boston)

# or ...
attach(Boston)
lm.fit = lm(medv ~ lstat)

# Continue
lm.fit
summary(lm.fit)
names(lm.fit)
coef(lm.fit)
confint(lm.fit)

# Predict - give new data and get new values (confidence or prediction interval)
predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "confidence")
predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "prediction")
plot(lstat, medv)
abline(lm.fit, lwd = 3, col = "red")
plot(lstat, medv, col = "red")
plot(lstat, medv, pch = 20)

# Plot with custom 'pch' character
plot(lstat, medv, pch = "+")
plot(1:20, 1:20, pch = 1:20)

# Multiple plots
par(mfrow = c(2, 2))
plot(lm.fit)
plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))

# Multiple Linear Regression
lm.fit = lm(medv ~ lstat + age, data = Boston)
summary(lm.fit)
lm.fit = lm(medv ~ ., data = Boston) # the '.' means against everything else
summary(lm.fit)
library(car)
vif(lm.fit)
lm.fit1 = lm(medv ~ . - age, data = Boston)
summary(lm.fit1)
lm.fit1 = update(lm.fit, ~ . - age)

# Interaction Terms
summary(lm(medv ~ lstat:age, data = Boston))
summary(lm(medv ~ lstat * age, data = Boston))

# Non-linear Transformations of the Predictors
lm.fit2 = lm(medv ~ lstat + I(lstat^2)) #'I' protects lsat squared in the regression
summary(lm.fit2)
lm.fit = lm(medv ~ lstat)
anova(lm.fit, lm.fit2)
par(mfrow = c(2, 2))
plot(lm.fit2)
lm.fit5 = lm(medv ~ poly(lstat, 5)) # use polymomials
summary(lm.fit5)
summary(lm(medv ~ log(rm), data = Boston))

# Qualitative (categorical) Predictors
fix(Carseats)
names(Carseats)
lm.fit = lm(Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
summary(lm.fit) # Beware of dummy variables coded by R for categorical data (alphabetical)
attach(Carseats)
contrasts(ShelveLoc) # Help view categorical data

# Writing User-defined Functions
LoadLibraries = function(){
 library(ISLR)
 library(MASS)
 print("The libraries have been loaded.")
 }
LoadLibraries()
```

- Least squares regression uses training observations (training data/set)
- training error - residuals - how well does our model fit the training data
    - training error can be measured using MSE (mean squared error)
    - this is similar to the R^2 (proportion of explained variance)
    - MSE and R^2 always improve with more covariates
    - if you give me infinite covariates, no matter how crazy (or nonsensical), I can get a perfect fit
      - but in this case it wont work well outside of the training set
- we really care about the model performance on a test set/data.
- when n<p, you can get an exact fit (for example, when n = p+1), but the test error will be awful
  - its always possible fit a polynomial with n-1 degrees to get a perfect fit
  - but our goal is to get a model that performs well on test data
  
```R
# Example code to fit model on training set and verify on test set
xtr <- matrix(rnorm(100 * 100), ncol = 100)
xte <- matrix(rnorm(100000 * 100), ncol = 100)
beta <- c(rep(1, 10), rep(0, 90))
ytr <- xtr%*%beta + rnorm(100)
yte <- xte%*%beta + rnorm(100000)
rsq <- trainerr <- testerr <- NULL
for(i in 2:100){
mod <- lm(ytr ~ xtr[, 1:i])
rsq <- c(rsq, summary(mod)$r.squared)
beta <- mod$coef[-1]
intercept <- mod$coef[1]
trainerr <- c(trainerr, mean((xtr[, 1:i]%*%beta + intercept - ytr)^2))
testerr <- c(testerr, mean((xte[, 1:i]%*%beta + intercept - yte)^2))
}
par(mfrow = c(1, 3))
plot(2:100, rsq, xlab = 'Number of Variables', ylab = "R Squared", log = "y"); abline(v = 10, col = "red")
plot(2:100, trainerr, xlab = 'Number of Variables', ylab = "Training Error", log = "y"); abline(v = 10, col = "red")
plot(2:100, testerr, xlab = 'Number of Variables', ylab = "Test Error", log = "y"); abline(v = 10, col = "red")
```

Bias and variance tradeoff
- bias is the difference between the true beta and the estimated (predicted) beta
  - the bias decreases with each added variable
  - but the variance in estimated betas keeps increasing across different experiments
  - Test Error = Bias^2 + Variance

How to split data into training and test data?
- our goal is to estimate test error
  - simply split into a validation set
    - be careful to split before the validation set can be used for anything! Set it aside first!
    - choose the number of covariates that minimizes the test/validation set error.
  - leave-one out cross-validation
    - repeat fit many times, and use n-1 for training, and the remaining sample as the validation data
    - the sum of the errors for each sample when used as the validation sample is the CV error
    - this estimates the optimal complexity (number of covariates), which afterwards the full dataset will be refitted with this new number of covariates
  - K-fold cross-validation
    - somehwere in the middle between the other two.
    - divide the data into *K* folds, then evaluate test error on each of the K folds.
    - bins are non-overlapping, often sorted in a non-random order then split into K folds

```R
# K-fold CV error estimation in R
library(boot)
xtr <- matrix(rnorm(100*100), ncol = 100)
beta <- c(rep(1, 10), rep(0, 90))
ytr <- xtr%*%beta + rnorm(100)
cv.err <- NULL
for(i in 2:50){
dat <- data.frame(x = xtr[, 1:i], y = ytr)
mod <- glm(y~., data = dat)
cv.err <- c(cv.err, cv.glm(dat, mod, K = 6)$delta[1])
}
plot(2:50, cv.err, xlab = "Number of Variables",
ylab = "6-Fold CV Error", log = "y")
abline(v = 10, col = "red")
```

### Thursday July 19, 2018 8:30a - 5:00p
```R
# Chaper 5 Lab: Cross-Validation and the Bootstrap
# The Validation Set Approach
library(ISLR)
set.seed(1) # reproduce the randomness
train = sample(392, 196) # will be indices to randomly split into training data
lm.fit = lm(mpg ~ horsepower, data = Auto, subset = train)
attach(Auto)

# Get MSE from model
mean((mpg - predict(lm.fit, Auto))[-train]^2)

# Fit a quadratic function to training data and get MSE on test data
lm.fit2 = lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)

# Fit third degree polynomial and get MSE on test
lm.fit3 = lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)

# Repeat with a new training set... shows degree of variability across folds
set.seed(2)
train = sample(392, 196)
lm.fit = lm(mpg ~ horsepower, subset = train)
mean((mpg - predict(lm.fit, Auto))[-train]^2)
lm.fit2 = lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((mpg - predict(lm.fit2, Auto))[-train]^2)
lm.fit3 = lm(mpg ~ poly(horsepower, 3),data = Auto,subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)

# Leave-One-Out Cross-Validation
glm.fit = glm(mpg ~ horsepower, data = Auto) # use glm when data are not continuous (e.g., logistic or poisson)
coef(glm.fit)
lm.fit = lm(mpg ~ horsepower, data = Auto)
coef(lm.fit)
library(boot)
glm.fit = glm(mpg ~ horsepower, data = Auto)
cv.err = cv.glm(Auto, glm.fit)
cv.err$delta # similar result to split sample CV

# Repeat leave one out CV 5 times for increasing polynomials
cv.error = rep(0, 5)
for (i in 1:5){
 glm.fit = glm(mpg ~ poly(horsepower, i), data = Auto)
 cv.error[i] = cv.glm(Auto, glm.fit)$delta[1]
 }
cv.error

# k-Fold Cross-Validation
set.seed(17)
cv.error.10 = rep(0, 10)
for (i in 1:10){
 glm.fit = glm(mpg ~ poly(horsepower, i), data = Auto)
 cv.error.10[i] = cv.glm(Auto, glm.fit, K = 10)$delta[1]
 }
cv.error.10
# This tunes the polynomial to use, then we build a final  model using this polynomial on the full dataset

# The Bootstrap
alpha.fn = function(data, index){
 X = data$X[index]
 Y = data$Y[index]
 return((var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y)))
 }
 
alpha.fn(Portfolio, 1:100)
set.seed(1)
alpha.fn(Portfolio, sample(100, 100, replace = T))
boot(Portfolio, alpha.fn, R = 1000)

# Estimating the Accuracy of a Linear Regression Model
boot.fn = function(data, index)
 return(coef(lm(mpg ~ horsepower, data = data, subset = index)))
boot.fn(Auto, 1:392)
set.seed(1)
boot.fn(Auto, sample(392, 392, replace = T))
boot.fn(Auto, sample(392, 392, replace = T))
boot(Auto, boot.fn, 1000)
summary(lm(mpg ~ horsepower, data = Auto))$coef
boot.fn = function(data, index)
 coefficients(lm(mpg ~ horsepower + I(horsepower^2), data = data, subset = index))
set.seed(1)
boot(Auto, boot.fn, 1000)
summary(lm(mpg ~ horsepower + I(horsepower^2), data = Auto))$coef
```
- in high dimensional data, we need to fit a less complex model (i.e., fewer variables)
- we use alternatives to least squares that allow us to choose the level of complexity:
  - variable pre-selection
  - forward stepwise regression
  - ridge regressio 
  - lasso regression
  - Principle components regression
- we select the complexity using a CV approach
- We need 'signal' variables and not 'noise' variables
- "Common mistakes are simple, and simple mistakes are common."
- for CV, split on independent units and not necessarily 'observations'

### High Dimensional data
- We want to make sure we do well in future test data
- we will accept some bias in order to reduce variance
- __Variable pre-selection__
  - choose a smaller set of *q* variables that are most correlated with the response
  - use least ssquares on a model with these q variables
    - simple and straightforward
```R
# Build toy data
xtr <- matrix(rnorm(100 * 100), ncol = 100)
beta <- c(rep(1, 10), rep(0, 90))
ytr <- xtr%*%beta + rnorm(100) # matrix multiplication

# Do correlations
cors <- cor(xtr, ytr)

# Select correlations that are good and build model
whichers <- which(abs(cors) > 0.2)
mod <- lm(ytr ~ xtr[, whichers])
print(summary(mod))
```
- each fold gives a different set of *q* features
- ONLY PRESELECT Q FROM CORRELATIONS WITH TRAINING DATA AND NOT FULL DATA
- goal is to pick the *q* variables that best predict the response
- __Best subset selection__
  - in other words, consider all possible 2^p models
  - can quickly become computationally intractable
- __Forward stepwise regression__ efficiently sorts through all these models
  - create all univariate models and pick the best predictor
  - fit all models with the first predictor and a second one... pick the best... repeat.
  - results in a nested set of models
  - not guaranteed to give you the best model of *q* variables, but rather a *good* set

```R
# Forward stepwise regression
xtr <- matrix(rnorm(100 * 100), ncol = 100)
beta <- c(rep(1, 10), rep(0, 90))
ytr <- xtr%*%beta + rnorm(100)
library(leaps)
out <- regsubsets(xtr, ytr, nvmax = 30, method = "forward")
print(summary(out))
print(coef(out, 1:10))
```

```R
# Chapter 6 Lab 1: Subset Selection Methods
# Best Subset Selection
library(ISLR)
   # fix(Hitters)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters = na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))

# Build All subset regression against all other features (".")
library(leaps)
regfit.full = regsubsets(Salary ~ ., Hitters)
summary(regfit.full)

# Build models with a max of 19 subsets
regfit.full = regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
reg.summary = summary(regfit.full)
names(reg.summary)
reg.summary$rsq

# Plot results
par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
which.max(reg.summary$adjr2)
points(11, reg.summary$adjr2[11], col = "red", cex = 2, pch = 20)
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = 'l')
which.min(reg.summary$cp)
points(10, reg.summary$cp[10], col = "red", cex = 2, pch = 20)
which.min(reg.summary$bic)
plot(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC", type = 'l')
points(6, reg.summary$bic[6], col = "red", cex = 2, pch = 20)
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
coef(regfit.full, 6)

# Forward and Backward Stepwise Selection
regfit.fwd = regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
summary(regfit.fwd)
regfit.bwd = regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "backward")
summary(regfit.bwd)
coef(regfit.full, 7)
coef(regfit.fwd, 7)
coef(regfit.bwd, 7)

# Choosing Among Models
set.seed(1)
train = sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)
test = (!train)
regfit.best = regsubsets(Salary ~ ., data = Hitters[train,], nvmax = 19)
test.mat = model.matrix(Salary ~ ., data = Hitters[test, ])
val.errors = rep(NA, 19)
for(i in 1:19){
   coefi = coef(regfit.best, id = i)
   pred = test.mat[, names(coefi)]%*%coefi
   val.errors[i] = mean((Hitters$Salary[test] - pred)^2)
}
val.errors
which.min(val.errors)
coef(regfit.best, 10)
predict.regsubsets = function(object, newdata, id, ...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  xvars = names(coefi)
  mat[, xvars]%*%coefi
  }
regfit.best = regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
coef(regfit.best, 10)
k = 10
set.seed(1)
folds = sample(1:k, nrow(Hitters), replace = TRUE)
cv.errors = matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))
for(j in 1:k){
  best.fit = regsubsets(Salary ~ ., data = Hitters[folds != j, ], nvmax = 19)
  for(i in 1:19){
    pred = predict(best.fit, Hitters[folds == j, ], id = i)
    cv.errors[j, i] = mean((Hitters$Salary[folds == j] - pred)^2)
    }
  }
mean.cv.errors = apply(cv.errors, 2, mean)
mean.cv.errors
par(mfrow = c(1, 1))
plot(mean.cv.errors, type = 'b')
reg.best = regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
coef(reg.best, 11)
```
- problem with backward selection is that you cannot fit a model with more features than observations
- __Ridge__ and __Lasso__ regression control complexity by NOT using least squares and instead by shrinking the regressino coefficients
  - this is called *regularization* or *penalization*
  - results from correlated variables, which when p is large, results in crazy coefficients for least squares which results is poor test error
    - too much model complexity
  - choose lambda by cross-valdiation
    - when lambda = 0, this is least squares
    - when lambda is huge, returns smaller beta coefficients
    - no feature selection with ridge regression
```R
# Ridge regression code
xtr <- matrix(rnorm(100 * 100), ncol = 100)
beta <- c(rep(1, 10), rep(0, 90))
ytr <- xtr%*%beta + rnorm(100)
library(glmnet)
cv.out <- cv.glmnet(xtr, ytr, alpha = 0, nfolds = 5)
print(cv.out$cvm)
plot(cv.out)
cat("CV Errors", cv.out$cvm, fill = TRUE)
cat("Lambda with smallest CV Error",
cv.out$lambda[which.min(cv.out$cvm)], fill = TRUE)
cat("Coefficients", as.numeric(coef(cv.out)), fill = TRUE)
cat("Number of Zero Coefficients",
sum(abs(coef(cv.out)) < 1e-8), fill = TRUE)
```

- The __Lasso__ regression tweaks the ridge regression to make most features zero.
  - Thus, this includes feature selection
  - the lasso penalizes complexity in two ways, using shrinkage (reducing betas to zero) and also removing features.
  - for lasso or ridge, all coefficients need to be standardized!!!! (performed by the glmnet package)
  - don't penalize the intercept
- penalized regressions also have only a single, global minimum, no local minima.
```R
# Lasso code - similar to ridge regression code
xtr <- matrix(rnorm(100 * 100), ncol = 100)
beta <- c(rep(1, 10), rep(0, 90))
ytr <- xtr%*%beta + rnorm(100)
library(glmnet)
cv.out <- cv.glmnet(xtr, ytr, alpha = 1, nfolds = 5)
print(cv.out$cvm)
plot(cv.out)
cat("CV Errors", cv.out$cvm, fill = TRUE)
cat("Lambda with smallest CV Error",
cv.out$lambda[which.min(cv.out$cvm)], fill = TRUE)
cat("Coefficients", as.numeric(coef(cv.out)), fill = TRUE)
cat("Number of Zero Coefficients", sum(abs(coef(cv.out)) < 1e-8), fill = TRUE)
```
```R
# Chapter 6 Lab 2: Ridge Regression and the Lasso
x = model.matrix(Salary ~ ., Hitters)[, -1]
y = Hitters$Salary

# Ridge Regression
library(glmnet)
grid = 10^seq(10, -2, length = 100)
ridge.mod = glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge.mod))
ridge.mod$lambda[50]
coef(ridge.mod)[, 50]
sqrt(sum(coef(ridge.mod)[-1,50]^2)) # a measure of model complexity
ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2)) # as penalty decreases, complexity increases
predict(ridge.mod,s=50,type="coefficients")[1:20,]
set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)
ridge.pred=predict(ridge.mod,s=4,newx=x[test,]) # s=4 means a lambda of 4
mean((ridge.pred-y.test)^2)
mean((mean(y[train])-y.test)^2)
ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,]) #get large lambda value, which removes almost all coefficents, so result should be similar to the intercept model above
mean((ridge.pred-y.test)^2)
ridge.pred=predict(ridge.mod,s=0,newx=x[test,],exact=T) # becomes least squares model
mean((ridge.pred-y.test)^2)
lm(y~x, subset=train)
predict(ridge.mod,s=0,exact=T,type="coefficients")[1:20,]
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
mean((ridge.pred-y.test)^2)
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,]

# The Lasso
lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
mean((lasso.pred-y.test)^2)
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]
```

- __Principle components regression__
  - finds a low-dimensional subspace of the data and then fits a model on that low-dimensional subspace, using least squares
  - make new lower dimensional datasets
  - do principle components (PC) on X, then do least squares on the principle components
  - each new PC is orthagonal (i.e., independent) to one another
  - use cross valdiation to choose M (the number of PCs to consider
  - use the line that best represents each PC
  - the last PC is already defined as whats left (no accounted for by the first p-1 PC)
  - doesn't do variable selection - all original predictors are included
  - coefficients are really connected to the interpretation, but still good at prediction
