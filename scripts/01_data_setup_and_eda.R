# Load required libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(randomForest)
library(xgboost)
library(caret)
library(car)

# Set seed for reproducibility
set.seed(123)

# Set working directory 
setwd("~/Documents/DATA522/House-Prices-Regression-Final-Project")

# Load the datasets from data/raw folder
train <- read_csv("data/raw/train.csv")
test <- read_csv("data/raw/test.csv")

# Initial EDA: Distribution of SalePrice before transformation
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(fill = "skyblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Sale Price")

# Log transform SalePrice (do this before any other transformations)
train$SalePrice <- log(train$SalePrice)

# Plot log-transformed SalePrice
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(fill = "seagreen", bins = 30) +
  theme_minimal() +
  labs(title = "Log-Transformed Distribution of SalePrice")

# Data Preparation Function
prepare_data <- function(df) {
  # Handle missing values
  df <- df %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(across(where(is.character), ~ifelse(is.na(.), "None", .))) %>%
    # Feature engineering
    mutate(
      HouseAge = YrSold - YearBuilt,
      RemodAge = YrSold - YearRemodAdd,
      TotalSF = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF`,
      Bathrooms = FullBath + 0.5 * HalfBath,
      TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + `3SsnPorch` + ScreenPorch
    ) %>%
    mutate(across(where(is.character), factor))
  return(df)
}

# Prepare training and test data
train <- prepare_data(train)
test <- prepare_data(test)

# Create training/validation split
split <- createDataPartition(train$SalePrice, p = 0.8, list = FALSE)
training_set <- train[split, ]
validation_set <- train[-split, ]

# Remove highly correlated numeric variables
numeric_vars <- training_set %>% select(where(is.numeric)) %>% names()
correlation_matrix <- cor(training_set[numeric_vars], use = "complete.obs")
high_correlations <- which(abs(correlation_matrix) > 0.8 & correlation_matrix != 1, arr.ind = TRUE)
if(length(high_correlations) > 0) {
  vars_to_remove <- unique(rownames(correlation_matrix)[high_correlations[,1]])
  vars_to_remove <- setdiff(vars_to_remove, "SalePrice")
  vars_to_remove <- vars_to_remove[vars_to_remove %in% names(training_set)]
  training_set <- training_set %>% select(-all_of(vars_to_remove))
  validation_set <- validation_set %>% select(-all_of(vars_to_remove))
  test <- test %>% select(-all_of(vars_to_remove))
  if(length(vars_to_remove) > 0) print(vars_to_remove)
}

# Remove zero-variance predictors
nzv <- nearZeroVar(training_set, saveMetrics = TRUE)
if (any(nzv$zeroVar)) {
  training_set <- training_set[, !nzv$zeroVar]
  validation_set <- validation_set[, names(training_set)]
  test <- test[, names(training_set)]
}

# Remove MiscVal and MiscFeature if present
to_drop <- c("MiscVal", "MiscFeature")
to_drop <- to_drop[to_drop %in% names(training_set)]
if(length(to_drop) > 0) {
  training_set <- training_set %>% select(-all_of(to_drop))
  validation_set <- validation_set %>% select(-all_of(to_drop))
  test <- test %>% select(-all_of(to_drop))
}

# Remove factors with only one level (if any remain)
one_level_factors <- names(training_set)[sapply(training_set, function(x) is.factor(x) && length(unique(na.omit(x))) < 2)]
if(length(one_level_factors) > 0) {
  training_set <- training_set[, !names(training_set) %in% one_level_factors]
  validation_set <- validation_set[, !names(validation_set) %in% one_level_factors]
  test <- test[, !names(test) %in% one_level_factors]
}

# Remove aliased variables (perfect collinearity)
repeat {
  initial_model <- lm(SalePrice ~ ., data = training_set)
  aliased_vars <- rownames(alias(initial_model)$Complete)
  aliased_vars <- aliased_vars[aliased_vars %in% names(training_set)]
  if(length(aliased_vars) == 0) break
  training_set <- training_set %>% select(-all_of(aliased_vars))
  validation_set <- validation_set %>% select(-all_of(aliased_vars))
  test <- test %>% select(-all_of(aliased_vars))
}

# Align factor levels between training, validation, and test (final step before modeling)
clean_and_align_factors <- function(train, valid) {
  for (col in intersect(names(train), names(valid))) {
    if (is.factor(train[[col]])) {
      train[[col]] <- droplevels(train[[col]])
      allowed_lvls <- levels(train[[col]])
      val_char <- as.character(valid[[col]])
      val_char[!val_char %in% allowed_lvls] <- NA
      valid[[col]] <- factor(val_char, levels = allowed_lvls)
      valid[[col]] <- droplevels(valid[[col]])
    }
  }
  for (col in names(train)) {
    if (is.factor(train[[col]])) {
      train[[col]] <- droplevels(train[[col]])
    }
  }
  return(list(train = train, valid = valid))
}

cleaned <- clean_and_align_factors(training_set, validation_set)
training_set <- cleaned$train
validation_set <- cleaned$valid

cleaned_test <- clean_and_align_factors(training_set, test)
training_set <- cleaned_test$train # usually unchanged, but for safety
test <- cleaned_test$valid

colSums(is.na(training_set))

#Check what is causing the problem loop
initial_model <- lm(SalePrice ~ ., data = training_set)
na_coefs <- names(which(is.na(coef(initial_model))))
print(na_coefs)

# Remove 'CBlock' from Exterior2nd in all sets
level_to_remove <- "CBlock"
varname <- "Exterior2nd"

for (df_name in c("training_set", "validation_set", "test")) {
  df <- get(df_name)
  if (varname %in% names(df) && is.factor(df[[varname]])) {
    df[[varname]][df[[varname]] == level_to_remove] <- NA
    df[[varname]] <- droplevels(df[[varname]])
    assign(df_name, df, envir = .GlobalEnv)
  }
}

#Confirming that the na_coefs is empty!
initial_model <- lm(SalePrice ~ ., data = training_set)
na_coefs <- names(which(is.na(coef(initial_model))))
print(na_coefs)

# VIF diagnostics (optional)
vif_values <- vif(initial_model)
print(vif_values)

#Re-align factors so I dont get the Error in model.frame.default(Terms, newdata, na.action = na.action, xlev = object$xlevels)
for (col in intersect(names(training_set), names(validation_set))) {
  if (is.factor(training_set[[col]])) {
    # Get the training set's levels
    allowed_lvls <- levels(training_set[[col]])
    # Force all values in validation_set not in allowed_lvls to NA
    validation_set[[col]][!validation_set[[col]] %in% allowed_lvls] <- NA
    # Reset the levels to be the same as training_set
    validation_set[[col]] <- factor(validation_set[[col]], levels = allowed_lvls)
    # Drop unused levels
    validation_set[[col]] <- droplevels(validation_set[[col]])
  }
}

#Same for my test set
for (col in intersect(names(training_set), names(test))) {
  if (is.factor(training_set[[col]])) {
    allowed_lvls <- levels(training_set[[col]])
    # Set any values not in allowed_lvls to NA
    test[[col]][!test[[col]] %in% allowed_lvls] <- NA
    # Reset the factor levels to be exactly those in training_set
    test[[col]] <- factor(test[[col]], levels = allowed_lvls)
    # Drop unused levels
    test[[col]] <- droplevels(test[[col]])
  }
}

# Predict on validation set and compute RMSE
val_preds <- predict(initial_model, newdata = validation_set)
val_preds_exp <- exp(val_preds)
actual_exp <- exp(validation_set$SalePrice)
rmse <- sqrt(mean((val_preds_exp - actual_exp)^2))
cat("Validation RMSE:", rmse, "\n")

#Still getting error in model.frame.default
table(validation_set$Foundation, useNA = "always")
levels(training_set$Foundation)

#Trying to figure out the error. Checking which levels are actually present in the training data
table(training_set$Foundation, useNA = "always")

#Slab is not a new level. So trying something else.
for (col in intersect(names(training_set), names(validation_set))) {
  if (is.factor(training_set[[col]])) {
    validation_set[[col]] <- factor(validation_set[[col]], levels = levels(training_set[[col]]))
  }
}

#Same for my test set
for (col in intersect(names(training_set), names(test))) {
  if (is.factor(training_set[[col]])) {
    test[[col]] <- factor(test[[col]], levels = levels(training_set[[col]]))
  }
}

#Refitting just in case.
initial_model <- lm(SalePrice ~ ., data = training_set)

# Try Predict on validation set and compute RMSE again
val_preds <- predict(initial_model, newdata = validation_set)
val_preds_exp <- exp(val_preds)
actual_exp <- exp(validation_set$SalePrice)
rmse <- sqrt(mean((val_preds_exp - actual_exp)^2))
cat("Validation RMSE:", rmse, "\n")

#Same issue so printing any factor values not in training_set
tryCatch({
  val_preds <- predict(initial_model, newdata = validation_set)
}, error = function(e) {
  print(e)
  for (col in names(validation_set)) {
    cat(col, ":", setdiff(unique(validation_set[[col]]), levels(training_set[[col]])), "\n")
  }
})


# Remove aliased columns from training, validation, and test sets
aliased <- na_coefs # or manually assign the vector of names
training_set <- training_set[ , !(names(training_set) %in% aliased)]
validation_set <- validation_set[ , !(names(validation_set) %in% aliased)]
test <- test[ , !(names(test) %in% aliased)]

#Refitting just in case.
initial_model <- lm(SalePrice ~ ., data = training_set)

# Try Predict on validation set and compute RMSE again
val_preds <- predict(initial_model, newdata = validation_set)
val_preds_exp <- exp(val_preds)
actual_exp <- exp(validation_set$SalePrice)
rmse <- sqrt(mean((val_preds_exp - actual_exp)^2))
cat("Validation RMSE:", rmse, "\n")

# Prepare test data for prediction
test_pred_input <- test %>% select(names(training_set))
test_pred_input <- test_pred_input[, !names(test_pred_input) %in% "SalePrice"]
predictions <- exp(predict(initial_model, newdata = test_pred_input))

# Submission
submission <- data.frame(
  Id = test$Id,
  SalePrice = predictions
)
write.csv(submission, "submission.csv", row.names = FALSE)

# Print model summary
cat("\nModel Performance:\n")
cat("R-squared:", summary(initial_model)$r.squared, "\n")
cat("Adjusted R-squared:", summary(initial_model)$adj.r.squared, "\n")
cat("Adjusted R-squared:", summary(initial_model)$adj.r.squared, "\n")
