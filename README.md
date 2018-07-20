# Summer Institute in Statistics for Big Data Workshop: Supervised Machine Learning
## Seattle, WA - July 18-20, 2018
Notes and R code
### don't forget to check out the 'caret' R package for ALL machine learning algorithms

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
train = sample(1:nrow(x), nrow(x) / 2)
test = (-train)
y.test = y[test]
ridge.mod = glmnet(x[train, ], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
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
lasso.mod = glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)
set.seed(1)
cv.out = cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.pred = predict(lasso.mod, s= bestlam, newx = x[test, ])
mean((lasso.pred - y.test)^2)
out = glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef = predict(out, type = "coefficients", s = bestlam)[1:20, ]
lasso.coef
lasso.coef[lasso.coef! = 0]
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

```R
# PC Regression using the PLS package
xtr <- matrix(rnorm(100 * 100), ncol = 100)
beta <- c(rep(1, 10), rep(0, 90))
ytr <- xtr%*%beta + rnorm(100)
library(pls)
out <- pcr(ytr ~ xtr, scale = TRUE, validation = "CV")
summary(out)
validationplot(out, val.type = "MSEP")

# or
# Principal Components Regression
library(pls)
set.seed(2)
pcr.fit = pcr(Salary ~ ., data = Hitters, scale = TRUE, validation = "CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP")

# Now fit using jus the training data
set.seed(1)
pcr.fit = pcr(Salary ~ ., data = Hitters, subset = train, scale = TRUE, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")
pcr.pred = predict(pcr.fit, x[test, ], ncomp = 7)
mean((pcr.pred - y.test)^2)
pcr.fit = pcr(y ~ x, scale = TRUE, ncomp = 7)
summary(pcr.fit)
```
- see Zhuang et al., BMC Bioinformatics, 2012 for a good example

### Classification
- predicting a categorical or qualitative response
  - e.g., cancer vs normal
  - e.g., tumor type 1 vs tumor type 2 vs tumor type 3
  - most commonly encountered in biomedical applications
  - often unordered categories
  - interested in the probability of belonging to each category
- we will discuss
  - k-nearest neighbors (KNN)
    - non-parametric, just look at the neighbors.  Model-free approach
    - *k* is the tuning parameter, chosen by Cross-Validation
    - results in a 'decision boundary' (see plots)
    - as k becomes larger, themodel become less complex
    - works when p is small... do not use in high dimensions!
      - could do a PC reduction first then try KNN
      - however, this requires two training parameters
  - logistic regression
    - extension of simple linear regression to a classification setting
    - Y is either 0 or 1 (categorical)
    - logistic regression calculates logi (log odds), so naturally includes probabilities
    - easily extended to multiple covariates
 ```R
# Logistic regression
xtr <- matrix(rnorm(1000 * 20), ncol = 20)
beta <- c(rep(1, 10), rep(0, 10))
ytr <- 1 * ((xtr%*%beta + 0.2 * rnorm(1000)) >= 0)
mod <- glm(ytr ~ xtr, family = "binomial")
print(summary(mod))
```
  - Five ways to extend logistic regression to high dimensions
    - variable pre-selection
    - forward stepwise logistic regression
    - Ridge logistic regression
    - LASSO logistic regression
    - PC logistic regression
  - Use the missclassification rate, predictive log likelihood, ROC or AUC metrics
```R
# LASSO logistic regression
xtr <- matrix(rnorm(1000 * 20), ncol = 20)
beta <- c(rep(1, 5), rep(0, 15))
ytr <- 1 * ((xtr%*%beta + 0.5 * rnorm(1000)) >= 0)
cv.out <- cv.glmnet(xtr, ytr, family = "binomial", alpha = 1)
plot(cv.out)
```

```R
# Chapter 4 Lab: Logistic Regression, LDA, QDA, and KNN
# The Stock Market Data
library(ISLR)
names(Smarket)
dim(Smarket)
summary(Smarket)
pairs(Smarket)
cor(Smarket)
cor(Smarket[,-9])
attach(Smarket)
plot(Volume)

# Logistic Regression
glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)
summary(glm.fits)
coef(glm.fits)
summary(glm.fits)$coef
summary(glm.fits)$coef[,4]
glm.probs=predict(glm.fits,type="response")
glm.probs[1:10]
contrasts(Direction)
glm.pred=rep("Down",1250)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction)
(507+145)/1250
mean(glm.pred==Direction)
train=(Year<2005)
Smarket.2005=Smarket[!train,]
dim(Smarket.2005)
Direction.2005=Direction[!train]
glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fits,Smarket.2005,type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)
mean(glm.pred!=Direction.2005)
glm.fits=glm(Direction~Lag1+Lag2,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fits,Smarket.2005,type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)
106/(106+76)
predict(glm.fits,newdata=data.frame(Lag1=c(1.2,1.5),Lag2=c(1.1,-0.8)),type="response")
```

```R
# LASSO fit
xtr = as.matrix(Smarket[train, -c(7,9)])
ytr = Smarket[train, 9]
xts = as.matrix(Smarket[!train, -c(7,9)])
yts = Smarket[!train, 9]
lasso.cv = cv.glmnet(x = xtr, y = ytr, family = "binomial", alpha = 1)

# Make predictions from test set using new fitted lambda
names(lasso.fit)
lasso.fit = cv.glmnet(x = xtr, y = ytr, family = "binomial", alpha = 1, lambda = lasso.cv$lambda)
lasso.pred = predict(lasso.cv, newx = xts, s = lasso.cv$lambda.1se, type = "response")
lasso.pred01 = as.factor(1*(lasso.pred > 0.5))
table(lasso.pred01, yts)
mean(lasso.pred01 == yts)
```
```R
# K-Nearest Neighbors
library(class)
train.X=cbind(Lag1,Lag2)[train,]
test.X=cbind(Lag1,Lag2)[!train,]
train.Direction=Direction[train]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.2005)
(83+43)/252
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)
mean(knn.pred==Direction.2005)

```
- discriminant analyses work especially well with >2 categories

### Friday July 20, 2018 8:30a - 5:00p

```R
# Linear Discriminant Analysis
library(MASS)
lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset=train)
lda.fit
plot(lda.fit)
lda.pred=predict(lda.fit, Smarket.2005)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class,Direction.2005)
mean(lda.class==Direction.2005)
sum(lda.pred$posterior[,1]>=.5)
sum(lda.pred$posterior[,1]<.5)
lda.pred$posterior[1:20,1]
lda.class[1:20]
sum(lda.pred$posterior[,1]>.9)

# K-Nearest Neighbors
library(class)
train.X=cbind(Lag1,Lag2)[train,]
test.X=cbind(Lag1,Lag2)[!train,]
train.Direction=Direction[train]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.2005)
(83+43)/252
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)
mean(knn.pred==Direction.2005)
```

- All discriminant analysis (DA) techniques use all the features
- the main difference between them are the ways the covriance matrices are estimated
- DDA is the best method in very high dimensions (don't use LDA/QDA)
- R package 'penalizedLDA

### Support Vector Machines (SVM)
- Fundamentally and numerically similar to logistic regression
- draw a line/plane in each dimension that separates the two classes
- SVM chooses the line/plane that gives the *maximum* separation between the variables
- Margin allows you to tell ho wmuch error you can make
- the support vectors are the few observations that are on the margin
  - the margin is a line, which is a linear combination of betas (the separating plane is one of the largin lines)
- But it rarely possible to separate all variables
  - use a support vector classifier that allows for violations
    - margin = 1 / length(beta)
    - calculate the error of the observations that are in violations
    - trade-off between the margin with the error
    - sometimes even this can be too difficult
  - in this case, a support vector machine changes it to a 3D problem using non-linear kernels
  - a non-linear kernel is like drawing a circle to be the separating line (with margins)
  - non-linear kernel is quite difficult in extremely high dimension
  
```R
# Chapter 9 Lab: Support Vector Machines

# Support Vector Classifier
set.seed(1)
x=matrix(rnorm(20*2), ncol=2)
y=c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1
plot(x, col=(3-y))
dat=data.frame(x=x, y=as.factor(y))
library(e1071)
svmfit=svm(y~., data=dat, kernel="linear", cost=10,scale=FALSE)
plot(svmfit, dat)
svmfit$index
summary(svmfit)
svmfit=svm(y~., data=dat, kernel="linear", cost=0.1,scale=FALSE)
plot(svmfit, dat)
svmfit$index
set.seed(1)
tune.out=tune(svm,y~.,data=dat,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)
xtest=matrix(rnorm(20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]=xtest[ytest==1,] + 1
testdat=data.frame(x=xtest, y=as.factor(ytest))
ypred=predict(bestmod,testdat)
table(predict=ypred, truth=testdat$y)
svmfit=svm(y~., data=dat, kernel="linear", cost=.01,scale=FALSE)
ypred=predict(svmfit,testdat)
table(predict=ypred, truth=testdat$y)
x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch=19)
dat=data.frame(x=x,y=as.factor(y))
svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
summary(svmfit)
plot(svmfit, dat)
svmfit=svm(y~., data=dat, kernel="linear", cost=1)
summary(svmfit)
plot(svmfit,dat)

# Support Vector Machine
set.seed(1)
x=matrix(rnorm(200*2), ncol=2)
x[1:100,]=x[1:100,]+2
x[101:150,]=x[101:150,]-2
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y))
plot(x, col=y)
train=sample(200,100)
svmfit=svm(y~., data=dat[train,], kernel="radial",  gamma=1, cost=1)
plot(svmfit, dat[train,])
summary(svmfit)
svmfit=svm(y~., data=dat[train,], kernel="radial",gamma=1,cost=1e5)
plot(svmfit,dat[train,])
set.seed(1)
tune.out=tune(svm, y~., data=dat[train,], kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary(tune.out)
table(true=dat[-train,"y"], pred=predict(tune.out$best.model,newdata=dat[-train,]))
```

### Batch effects
- Steps to reduce batch effects
  - randomize smaple run times (e.g., don't do controls first and cases second)
  - train classification on a mixed set of samples, e.g., across institutions and blinded
  - validate results on a independent set of samples from a different institution

### Decision trees
- different approach to non-linear modeling that is easily interpretable
- some loss of accuracy
- all the splits are binary splits
- regions of the tree are determined using recursive binary splitting
- p\*n initial splits to consider
- tuning parameter... how many splits (i.e., how big to grow the tree?)
  - large tree = low bias but high variance
  - small tree = high bias but low variance
  - common to grow a very large tree, then prune it down
    - cost-complexity pruning
    - alpha is a parameter that penalizes the number of nodes (i.e., complexity)
    - alpha = 0 is the *FULL* tree
    - larger alpha = increasingly pruned tree
    - looks for splits that minimize *node impurity*
    - cross-entropy and Gini index are used to grow a tree, and missclassification error is generally used to prune the tree
    - bagging and boosting can degrease the bias and variance, respectively
  - Bootstrap aggregation (bagging)
    - each bootstrap replicate is a new, randomly sampled list of n pairs of observations (X and Y) with replacement
    - this decreases the variance of our estimates at no cost really other than the computational time
      - example workflow:  1) build 10000 bootstrap datasets of n observations and associated predictors, 2) build decision trees for each, 3) take test samples and make a prediction for each of the 10000 trees, 4) average/majority vote across predictions.
      - bagging only really helps in very non-linear situations.  If linear, lots of bootstraps results in the orginal linear estimates
    - bagging, however, loses interpretability, which ws an original benefit of decision trees
  - __Random Forests__
    - particularly useful with trees
    - builds a large collection of de-correlated trees
    - very good predctive performance, but limited interpretability
    - goal is to get the pairwise corellation between trees smaller, which decreases the variance
    - choose m < p variables at every split in the tree
      - average the resulting B (# of variables) trees for regression
      - take majority vote for classicification
    - a great "out of the box" method
    - very little tuning
    - recommendations
      - For regression, take *m = p/3*, and minimum node size 5
      - For classification, take *m = p^1/2*, and minimum node size 1
    - often only a few hundred to thousand trees are needed
    - variable importance plots (how often a feature appears in various nodes)
    
```R
# Chapter 8 Lab: Decision Trees
# Fitting Classification Trees
library(tree)
library(ISLR)
attach(Carseats)
High=ifelse(Sales<=8,"No","Yes")
Carseats=data.frame(Carseats,High)
tree.carseats=tree(High~.-Sales,Carseats)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats,pretty=0)
tree.carseats # look at all splits
set.seed(2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(86+57)/200
set.seed(3)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass) # fits the whole range and we pick our parameters
names(cv.carseats)
cv.carseats
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")
prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(94+60)/200
prune.carseats=prune.misclass(tree.carseats,best=15)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(86+62)/200

# Fitting Regression Trees
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston,pretty=0)
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')
prune.boston=prune.tree(tree.boston,best=5)
plot(prune.boston)
text(prune.boston,pretty=0)
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

# Bagging and Random Forests
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
importance(rf.boston)
varImpPlot(rf.boston)
```

### Boosting
- random forest ONLY used with trees
- boosting used with both trees and other methods
- trees grown sequentially from only a single dataset
- small, simple trees, with each tree fitting well to small parts of the data
- three tuning parameters
  - B - the number of trees
  - lambda - the shrinkage parameter
  - d - the number of splits in each tree
    - often, d=1 (stunps) is good enough if everything is uncorrelated
    - d can also be though of as an interaction depth
  - lambda\*B is essentially the total number of trees to look at
  - boosting with too many trees will result in overfitting!

### Stacking
- a method to combine across models to have a better model
- can use cross validation



### Final Note:
- to bootstrap the predictive accuracy of a model, sample observations with replacement, then use the individuals that did not make it into the the bootstrap in the test set, to avoid have the same indibiduals in both the training and test set.
