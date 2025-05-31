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

# Initial EDA
# Look at the distribution of SalePrice before transformation
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
prepare_data <- function(df, is_training = TRUE) {
  # Handle missing values
  df <- df %>%
    mutate(across(where(is.numeric), 
                  ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(across(where(is.character), 
                  ~ifelse(is.na(.), "None", .)))
  
  # Feature engineering
  df <- df %>%
    mutate(
      HouseAge = YrSold - YearBuilt,
      RemodAge = YrSold - YearRemodAdd,
      TotalSF = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF`,
      Bathrooms = FullBath + 0.5 * HalfBath,
      TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + `3SsnPorch` + ScreenPorch
    )
  
  # Convert character columns to factors
  df <- df %>%
    mutate(across(where(is.character), factor))
  
  return(df)
}

# Prepare training and test data
train <- prepare_data(train, is_training = TRUE)
test <- prepare_data(test, is_training = FALSE)

# Create training/validation split
split <- createDataPartition(train$SalePrice, p = 0.8, list = FALSE)
training_set <- train[split, ]
validation_set <- train[-split, ]

# Clean factor levels function
clean_factor_data <- function(train_data, valid_data) {
  # Remove factors with only one level
  bad_factors <- names(train_data)[sapply(train_data, function(x) {
    is.factor(x) && length(unique(na.omit(x))) < 2
  })]
  
  train_data <- train_data[, !names(train_data) %in% bad_factors]
  valid_data <- valid_data[, !names(valid_data) %in% bad_factors]
  
  # Align factor levels
  for(col in names(train_data)) {
    if(is.factor(train_data[[col]])) {
      valid_data[[col]] <- factor(valid_data[[col]], 
                                  levels = levels(train_data[[col]]))
    }
  }
  
  return(list(train = train_data, valid = valid_data))
}

# Clean the data
clean_data <- clean_factor_data(training_set, validation_set)
training_set <- clean_data$train
validation_set <- clean_data$valid

# Check correlations
numeric_vars <- training_set %>%
  select_if(is.numeric) %>%
  names()

correlation_matrix <- cor(training_set[numeric_vars], use = "complete.obs")
high_correlations <- which(abs(correlation_matrix) > 0.8 & 
                             correlation_matrix != 1 & 
                             !grepl("SalePrice", rownames(correlation_matrix)), 
                           arr.ind = TRUE)

# Remove highly correlated variables
if(length(high_correlations) > 0) {
  vars_to_remove <- unique(rownames(correlation_matrix)[high_correlations[,1]])
  training_set <- training_set %>% select(-all_of(vars_to_remove))
  validation_set <- validation_set %>% select(-all_of(vars_to_remove))
}

# Find and remove perfect correlations
if(length(high_correlations) > 0) {
  vars_to_remove_perfect <- unique(rownames(correlation_matrix)[high_correlations[,1]])
  # Only remove variables that exist in the dataset
  vars_to_remove_perfect <- vars_to_remove_perfect[vars_to_remove_perfect %in% names(training_set)]
  if(length(vars_to_remove_perfect) > 0) {
    training_set <- training_set %>% select(-all_of(vars_to_remove_perfect))
    validation_set <- validation_set %>% select(-all_of(vars_to_remove_perfect))
  }
}

# Fit initial model with a more robust formula
initial_model <- lm(SalePrice ~ ., data = training_set)

# Use a different approach for VIF
library(car)
alias_model <- alias(initial_model)
if(!is.null(alias_model$Complete)) {
  # Get names of aliased variables
  aliased_vars <- rownames(alias_model$Complete)
  # Only remove variables that exist in the dataset
  aliased_vars <- aliased_vars[aliased_vars %in% names(training_set)]
  if(length(aliased_vars) > 0) {
    training_set <- training_set %>% select(-all_of(aliased_vars))
    validation_set <- validation_set %>% select(-all_of(aliased_vars))
    # Refit the model without aliased variables
    initial_model <- lm(SalePrice ~ ., data = training_set)
  }
}

# Now check VIF values
vif_values <- vif(initial_model)
high_vif_vars <- names(which(vif_values > 5))
# Only remove variables that exist in the dataset
high_vif_vars <- high_vif_vars[high_vif_vars %in% names(training_set)]

if(length(high_vif_vars) > 0) {
  training_set <- training_set %>% select(-all_of(high_vif_vars))
  validation_set <- validation_set %>% select(-all_of(high_vif_vars))
  final_model <- lm(SalePrice ~ ., data = training_set)
} else {
  final_model <- initial_model
}


# Evaluate model performance
predictions_val <- predict(final_model, newdata = validation_set)
rmse_val <- sqrt(mean((predictions_val - validation_set$SalePrice)^2))
cat("Validation RMSE:", rmse_val, "\n")

# Prepare test data for prediction
test <- test %>% select(names(training_set))
test <- test[, !names(test) %in% "SalePrice"]

# Make predictions on test set
predictions <- exp(predict(final_model, newdata = test))

# Create submission file
submission <- data.frame(
  Id = test$Id,
  SalePrice = predictions
)

# Write submission file
write.csv(submission, "submission.csv", row.names = FALSE)

# Print model summary and performance metrics
cat("\nModel Performance:\n")
cat("R-squared:", summary(final_model)$r.squared, "\n")
cat("Adjusted R-squared:", summary(final_model)$adj.r.squared, "\n")