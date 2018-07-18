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

```
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
```

