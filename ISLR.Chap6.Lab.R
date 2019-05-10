library(ISLR)
library(leaps)
library(glmnet)
library(pls)


# 6.5 Lab 1 subset selection
# summary data
View(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hit2 <- na.omit(Hitters)
summary(Hit2)
dim(Hit2)


fit_full <- regsubsets(Salary ~ ., data = Hit2)
summary(fit_full)

# best subset selection
fit_full <- regsubsets(Salary ~ ., data = Hit2, nvmax = 19)
fit_sum <- summary(fit_full)
names(fit_sum)
fit_sum$rsq

# plot estimates(R2,bic..) in feature selection
min_rss <- which.min(fit_sum$rss)
max_adjr2 <- which.max(fit_sum$adjr2)
min_cp <- which.min(fit_sum$cp)
min_bic <- which.min(fit_sum$bic)
par(mfrow = c(1,1))
plot(fit_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
points(min_rss, fit_sum$rss[min_rss], col = "red", cex = 2, pch = 20)
plot(fit_sum$adjr2, xlab = "Number of Variables", ylab = "adj_R2", type = "l")
points(max_adjr2, fit_sum$adjr2[max_adjr2], col = "red", cex = 2, pch = 20)
plot(fit_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
points(min_cp, fit_sum$cp[min_cp], col = "red", cex = 2, pch = 20)
plot(fit_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
points(min_bic, fit_sum$bic[min_bic], col = "red", cex = 2, pch = 20)

# coef of model with exact variable number,here is 7
plot(fit_full, scale = "bic")
coef(fit_full, id = 7)

# forward and backward subset selection
fit_fwd <- regsubsets(Salary ~ ., data = Hit2, nvmax = 19, method = "forward")
fit_bwd <- regsubsets(Salary ~ ., data = Hit2, nvmax = 19, method = "backward")
coef(fit_fwd, id = 7)
coef(fit_bwd, id = 7)

# validation set approach
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(Hit2), replace = TRUE)
test <- !train
fit_best <- regsubsets(Salary ~ ., data = Hit2[train,], nvmax = 19)
test_mat <- model.matrix(Salary ~ ., data = Hit2[test,])
val_err <- rep(NA, 19)
for (i in 1:19) {
  coef_i <- coef(fit_best, id = i)
  selected_feature <- names(coef_i)
  pred <- test_mat[, selected_feature] %*% coef_i
  mse <- mean((Hit2[test,]$Salary - pred)^2)
  val_err[i] <- mse
}
val_err
best_feature_number <- which.min(val_err)
coef(fit_best, id = 10)
coef(fit_full, id = 10)


# predict function for regsubsets
predict.regsubsets <- function(fm, newdata, id, ...){
  form <- as.formula(fm$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(fm, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}


# CV approach
k <- 10
set.seed(1)
folds <- sample(1:k, nrow(Hit2), replace = TRUE)
# split data into K equal parts
# n <- nrow(Hit2)
# k_vec <- rep(1:k, ceiling(n/k))[1:n]
# folds2 <- sample(k_vec, n)
cv_err_mat <- matrix(NA, k, 19, dimnames = list(paste(1:k), paste(1:19)))

for (kf in 1:k) {
  best_fit <- regsubsets(Salary ~ ., data = Hit2[folds!=kf,], nvmax = 19)
  for (var_num in 1:19) {
    pred <- predict.regsubsets(best_fit, Hit2[folds==kf,], var_num)
    cv_err_mat[kf, var_num] <- mean((pred - Hit2$Salary[folds==kf])^2)
  }
}
cv_err_mat
mean_cv_err <- apply(cv_err_mat, 2, mean)
plot(mean_cv_err, type = "b")
#points(which.min(mean_cv_err), min(mean_cv_err), col = "red")
reg_best <- regsubsets(Salary ~ ., data = Hit2, nvmax = 19)
coef(reg_best, which.min(mean_cv_err))


# 6.6 Lab2
# ridge and lasso regression

# predict_var to design matrix, remove Intercept
x <- model.matrix(Salary ~ ., Hit2)[, -1]
y <- Hit2$Salary
grid <- 10^seq(10, -2, length.out = 100)
ridge_mod <- glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge_mod))
ridge_mod$lambda[50]
coef(ridge_mod)[, 50]
sqrt(sum((coef(ridge_mod)[-1, 50])^2))
ridge_mod$lambda[60]
coef(ridge_mod)[, 60]
sqrt(sum((coef(ridge_mod)[-1, 60])^2))
# a new lambda`s coef
predict(ridge_mod, s = 50, type = "coefficients")[1:20, ]
# ridge validation-set approach
set.seed(1)
n <- nrow(Hit2)
train <- sample(1:n, n/2)
test <- (-train)
y_test <- y[test]

# MSE when lambda=4
ridge_mod <- glmnet(x[train,], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
ridge_pred <- predict(ridge_mod, s = 4, newx = x[test,])
mean((ridge_pred - y_test)^2)
# MSE when lambda=0, equal simple linear model
# if set "exact=TRUE", must supply original x and y
ridge_pred0 <- predict(ridge_mod, s = 0, newx = x[test,], exact = TRUE, x = x[train,], y = y[train])
mean((ridge_pred0 - y_test)^2)
# coef of lm and ridge(lamda=0)
lm(y ~ x, subset = train)
predict(ridge_mod, s = 0, type = "coefficients")[1:20,]

# CV approach to select lambda
set.seed(1)
cv_out <- cv.glmnet(x[train,], y[train], alpha = 0, nfolds = 10)
plot(cv_out)
best_lambda <- cv_out$lambda.min
ridge_pred_bestlam <- predict(ridge_mod, s = best_lambda, newx = x[test,])
mean((ridge_pred_bestlam - y[test])^2)

final_ridge_mod <- glmnet(x, y, alpha = 0)
predict(final_ridge_mod, s = best_lambda, type = "coefficients")[1:20,]


# lasso
lasso_mod <- glmnet(x[train,], y[train], alpha = 1, lambda = grid)
plot(lasso_mod)

set.seed(1)
cv_out <- cv.glmnet(x[train,], y[train], alpha = 1)
plot(cv_out)
best_lambda <- cv_out$lambda.min
lasso_pred <- predict(lasso_mod, s = best_lambda, newx = x[test,])
mean((lasso_pred - y[test])^2)

final_lasso_mod <- glmnet(x, y, alpha = 1, lambda = grid)
final_coef <- predict(final_lasso_mod, s = best_lambda, type = "coefficients")[1:20,]
sum(final_coef==0)


# PCR
set.seed(2)
pcr_fit <- pcr(Salary ~ ., data = Hit2, scale = TRUE, validation = "CV")
summary(pcr_fit)
validationplot(pcr_fit, val.type = "MSEP")

set.seed(1)
pcr_fit <- pcr(Salary ~ ., data = Hit2, subset = train, scale = TRUE, validation = "CV")
validationplot(pcr_fit, val.type = "MSEP")
pcr_pred <- predict(pcr_fit, x[test,], ncomp = 7)
mean((pcr_pred - y[test])^2)

final_pcr_fit <- pcr(Salary ~ ., data = Hit2, scale = TRUE, ncomp = 7)
summary(final_pcr_fit)
 
# PLS
set.seed(1)
pls_fit <- plsr(Salary ~ ., data = Hit2, subset = train, scale = TRUE, validation = "CV")
summary(pls_fit)
validationplot(pls_fit, val.type = "MSEP")
pls_pred <- predict(pls_fit, x[test,], ncomp = 2)
mean((pls_pred - y[test])^2)

final_pls_fit <- plsr(Salary ~ ., data = Hit2, scale = TRUE, ncomp = 2)
summary(final_pls_fit)














