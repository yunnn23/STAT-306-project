library(ggplot2)
library(GGally)
library(leaps)
library(tidyr)
library(dplyr)

# Read dataset
data <- read.csv("insurance.csv", header = TRUE)

# Insurance charge is our response variable
# Normality
hist(data$charges,xlab = "Insurance charges (USD)", main = "Histogram of Medical insurance charges")
# The histogram clearly shows that distribution of medical expenses is right-skewed

# Fit a linear model between insurance charge and bmi
model_1 <- lm(charges ~ bmi, data = data)
summary(model_1)
# The summary shows that one unit increase in the bmi is associated with a 393.87 change 
# in the medical expense

# Plot
ggplot(data, aes(x = bmi, y = charges)) +
  geom_point() +
  geom_smooth(method = "lm", col = "red") +
  labs(title = "Insurance Charges vs. Bmi", x = "bmi (kg/m^2)", y = "Insurance Charges ($)")


# Exploring relationship between response and other explanatory variables

# Box plot of Insurance charges by Sex
ggplot(data = data, aes(x = sex, y = charges, fill = sex)) +
  geom_boxplot() + 
  geom_point() +
  labs(title = "Medical Costs by Sex", x = "Sex", y = "Insurance Charges ($)")
# The median of medical costs is similar between female and male


# Box plot of Insurance charges by Smoking status
ggplot(data = data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot() + 
  geom_point() +
  labs(title = "Medical Expenses by Smoking Status", x = "Smoking", y = "Insurance Charges ($)")
# The box plot shows that Medical expenses among the smoking group is considerably higher than 
# that of non-smoking people


# Box plot of Insurance charges by Region
ggplot(data = data, aes(x = region, y = charges, fill = region)) +
  geom_boxplot() + 
  geom_point() +
  labs(title = "Medical Expenses by Region", x = "Region", y = "Insurance Charges ($)")
# The median of medical expenses appears fairly similar across all regions


# Comparing correlation by pair plot
ggpairs(data)


# Fit a linear model with all of the explanatory variables
# This is our additive model; model_2
model_2 <- lm(charges ~ bmi + age + sex + children + smoker + region, data = data)
summary(model_2)
# The bmi, age, number of children, smoking status, southeast and southwest region variables 
# seem to be statistically significant whereas the sex, northwest region do not appear
# to be significant

# Residual Plot
plot(model_2$fitted.values, model_2$residuals)
abline(0, 0)


# We'd like to explore interaction between bmi and other variable

# Fit a linear model with interaction term between Bmi and Sex 
# This is our interaction model; model_3
model_3 <- lm(charges ~ bmi + age + sex + children + smoker + region + (bmi*sex), 
              data = data)
summary(model_3)
# The interaction term between Bmi and Sex does not seem to be statistically significant
# We determine not to include the interaction

# Model Selection
model_selection_set <- regsubsets(charges ~ bmi + age + sex + children + smoker + region,
                                  data = data,
                                  nvmax = NULL)
summary(model_selection_set)$which

# Summary of model selection
summary_ms <- tibble(n_input_variables = 1:8,
                     RSQ = summary(model_selection_set)$rsq,
                     ADJ.R2 = summary(model_selection_set)$adjr2,
                     Cp = summary(model_selection_set)$cp,
                     BIC = summary(model_selection_set)$bic)
summary_ms
# We are selecting 5 input variables, 
# which are bmi, age, number of children, smoking status, and region

# Fit a linear model selected by model selection
# This is our reduced model
reduced_model <- lm(charges ~ bmi + age + children + smoker + region, data = data)
summary(reduced_model)
# Now, all variables except southwest region are statistically significant


# Fit a linear model with all variables
# This is our full model, which is the same with model_2 we created above
full_model <- lm(charges ~ bmi + age + sex + children + smoker + region, data = data)
summary(full_model)

# Breaking the dataset into training set and testing set with 60% and 40%, respectively
set.seed(123)
training_set <- sample_n(data, size = nrow(data) * 0.60, replace = FALSE)
testing_set <- anti_join(data, training_set)
summary(training_set)
summary(testing_set)

# RMSE of the full model
full_model_train <- lm(charges ~ bmi + age + sex + children + smoker + region, 
                       data = training_set)
actuals <- testing_set$charges
preds_full <- predict(full_model_train, testing_set)
rmse_full <- sqrt(mean((actuals - preds_full)^2))
rmse_full
# The residual mean square error of the full model is 5726.94


# RMSE of the reduced model (selected by model selection)
reduced_model_train <- lm(charges ~ bmi + age + children + smoker + region, 
                       data = training_set)
actuals <- testing_set$charges
preds_reduced <- predict(reduced_model_train, testing_set)
rmse_reduced <- sqrt(mean((actuals - preds_reduced)^2))
rmse_reduced
# The residual mean square error of the reduced model is 5725.864

# Based on RMSE provided, it suggests that the reduced model is better than the full model 


# VIF (Variance Inflation Factor)
vif(reduced_model_train)
# The VIF results indicate there is no concerning presence of multicollinearity
# since all values are small


# QQ-plot
plot(reduced_model_train, 2, main = "Reduced Model")

# Histogram of residuals
hist(residuals(object = reduced_model_train),
     main = "Histogram of Residuals for Reduced Model",
     xlab = "Residuals")

# QQ-plot and the histogram of residuals for Full Model shows that 
# the distribution is not normal and there is right skew in the residuals

# To improve the fit of the model by taking the log in Insurance Charges
reduced_log_model <- lm(log(charges) ~ bmi + age + children + smoker + region,
                        data = training_set)
summary(reduced_log_model)


# QQ-plot for log
plot(reduced_log_model, 2, main = "Reduced Log Model")

# Histogram for Residuals of log
hist(residuals(object = reduced_log_model),
     main = "Histogram of Residuals for Reduced Log Model",
     xlab = "Residuals")

## PCA
insurance_numerical <- subset(data, select = -c(sex, smoker, region,charges))
insurance_numerical <- scale(insurance_numerical)
pr_insurance <- princomp(insurance_numerical)
summary(pr_insurance)
pr_insurance$loadings[, 1:3]

