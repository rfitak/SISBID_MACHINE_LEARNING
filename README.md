# Summer Institute in Statistics for Big Data Workshop: Supervised Machine Learning
## Seattle, WA - July 18-20, 2018
Notes and R code

### Wednesday July 18, 2018 1:30p - 5:00 pm
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
x=seq(-pi,pi,length=50)
y=x

# Makes a new number from each combiantion of X and Y
f=outer(x,y,function(x,y)cos(y)/(1+x^2))

# Contour plot - like a topo map
contour(x,y,f)
contour(x,y,f,nlevels=45,add=T)
fa=(f-t(f))/2
contour(x,y,fa,nlevels=15)

# Image produce a plot similar to a heatmap
image(x,y,fa)

# 3D plot of same data
persp(x,y,fa)
persp(x,y,fa,theta=30)
persp(x,y,fa,theta=30,phi=20)
persp(x,y,fa,theta=30,phi=70)
persp(x,y,fa,theta=30,phi=40)

# 'identifiy'
identify(horsepower,mpg,name)
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
lm.fit=lm(medv~lstat,data=Boston)

# or ...
attach(Boston)
lm.fit=lm(medv~lstat)

# Continue
lm.fit
summary(lm.fit)
names(lm.fit)
coef(lm.fit)
confint(lm.fit)

# Predict - give new data and get new values (confidence or prediction interval)
predict(lm.fit,data.frame(lstat=(c(5,10,15))), interval="confidence")
predict(lm.fit,data.frame(lstat=(c(5,10,15))), interval="prediction")
plot(lstat,medv)
abline(lm.fit, lwd = 3, col = "red")
plot(lstat,medv,col="red")
plot(lstat,medv,pch=20)

# Plot with custom 'pch' character
plot(lstat,medv,pch="+")
plot(1:20,1:20,pch=1:20)

# Multiple plots
par(mfrow=c(2,2))
plot(lm.fit)
plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))

# Multiple Linear Regression
lm.fit=lm(medv~lstat+age,data=Boston)
summary(lm.fit)
lm.fit=lm(medv~.,data=Boston) # the '.' means against everything else
summary(lm.fit)
library(car)
vif(lm.fit)
lm.fit1=lm(medv~.-age,data=Boston)
summary(lm.fit1)
lm.fit1=update(lm.fit, ~.-age)

# Interaction Terms
summary(lm(medv~lstat:age,data=Boston))
summary(lm(medv~lstat*age,data=Boston))

# Non-linear Transformations of the Predictors
lm.fit2=lm(medv~lstat+I(lstat^2)) #'I' protects lsat squared in the regression
summary(lm.fit2)
lm.fit=lm(medv~lstat)
anova(lm.fit,lm.fit2)
par(mfrow=c(2,2))
plot(lm.fit2)
lm.fit5=lm(medv~poly(lstat,5)) # use polymomials
summary(lm.fit5)
summary(lm(medv~log(rm),data=Boston))

# Qualitative (categorical) Predictors
fix(Carseats)
names(Carseats)
lm.fit=lm(Sales~.+Income:Advertising+Price:Age,data=Carseats)
summary(lm.fit) # Beware of dummy variables coded by R for categorical data (alphabetical)
attach(Carseats)
contrasts(ShelveLoc) # Help view categorical data

# Writing User-defined Functions
LoadLibraries=function(){
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
xtr <- matrix(rnorm(100*100),ncol=100)
xte <- matrix(rnorm(100000*100),ncol=100)
beta <- c(rep(1,10),rep(0,90))
ytr <- xtr%*%beta + rnorm(100)
yte <- xte%*%beta + rnorm(100000)
rsq <- trainerr <- testerr <- NULL
for(i in 2:100){
mod <- lm(ytr~xtr[,1:i])
rsq <- c(rsq,summary(mod)$r.squared)
beta <- mod$coef[-1]
intercept <- mod$coef[1]
trainerr <- c(trainerr, mean((xtr[,1:i]%*%beta+intercept - ytr)^2))
testerr <- c(testerr, mean((xte[,1:i]%*%beta+intercept - yte)^2))
}
par(mfrow=c(1,3))
plot(2:100,rsq, xlab='Number of Variables', ylab="R Squared", log="y"); abline(v=10,col="red")
plot(2:100,trainerr, xlab='Number of Variables’', ylab="Training Error",log="y"); abline(v=10,col="red")
plot(2:100,testerr, xlab='Number of Variables', ylab="Test Error",log="y"); abline(v=10,col="red")
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
