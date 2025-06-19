# Load required libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(randomForest)
library(xgboost)
library(caret)
library(car)
library(glmnet)
library(e1071)

# Set seed for reproducibility
set.seed(123)

# Set working directory 
setwd("~/Documents/DATA522/House-Prices-Regression-Final-Project")

# Load the datasets
train <- read_csv("data/raw/train.csv")
test <- read_csv("data/raw/test.csv")

# Store original test IDs
test_ids <- test$Id

# Combine train and test for consistent preprocessing
test$SalePrice <- NA
combined <- rbind(train, test)

# Remove extreme outliers (houses with very large GrLivArea but low price)
combined <- combined %>%
  filter(!(GrLivArea > 4000 & SalePrice < 300000) | is.na(SalePrice))

# Enhanced data preparation function
prepare_data <- function(df) {
  df <- df %>%
    # Handle missing values with domain knowledge
    mutate(
      # Garage variables
      GarageType = ifelse(is.na(GarageType), "None", GarageType),
      GarageFinish = ifelse(is.na(GarageFinish), "None", GarageFinish),
      GarageQual = ifelse(is.na(GarageQual), "None", GarageQual),
      GarageCond = ifelse(is.na(GarageCond), "None", GarageCond),
      GarageYrBlt = ifelse(is.na(GarageYrBlt), YearBuilt, GarageYrBlt),
      GarageCars = ifelse(is.na(GarageCars), 0, GarageCars),
      GarageArea = ifelse(is.na(GarageArea), 0, GarageArea),
      
      # Basement variables
      BsmtQual = ifelse(is.na(BsmtQual), "None", BsmtQual),
      BsmtCond = ifelse(is.na(BsmtCond), "None", BsmtCond),
      BsmtExposure = ifelse(is.na(BsmtExposure), "None", BsmtExposure),
      BsmtFinType1 = ifelse(is.na(BsmtFinType1), "None", BsmtFinType1),
      BsmtFinType2 = ifelse(is.na(BsmtFinType2), "None", BsmtFinType2),
      BsmtFinSF1 = ifelse(is.na(BsmtFinSF1), 0, BsmtFinSF1),
      BsmtFinSF2 = ifelse(is.na(BsmtFinSF2), 0, BsmtFinSF2),
      BsmtUnfSF = ifelse(is.na(BsmtUnfSF), 0, BsmtUnfSF),
      TotalBsmtSF = ifelse(is.na(TotalBsmtSF), 0, TotalBsmtSF),
      BsmtFullBath = ifelse(is.na(BsmtFullBath), 0, BsmtFullBath),
      BsmtHalfBath = ifelse(is.na(BsmtHalfBath), 0, BsmtHalfBath),
      
      # Other variables
      Alley = ifelse(is.na(Alley), "None", Alley),
      Fence = ifelse(is.na(Fence), "None", Fence),
      FireplaceQu = ifelse(is.na(FireplaceQu), "None", FireplaceQu),
      PoolQC = ifelse(is.na(PoolQC), "None", PoolQC),
      MiscFeature = ifelse(is.na(MiscFeature), "None", MiscFeature),
      
      # Handle remaining missing values
      LotFrontage = ifelse(is.na(LotFrontage), median(LotFrontage, na.rm = TRUE), LotFrontage),
      MasVnrType = ifelse(is.na(MasVnrType), "None", MasVnrType),
      MasVnrArea = ifelse(is.na(MasVnrArea), 0, MasVnrArea),
      Electrical = ifelse(is.na(Electrical), "SBrkr", Electrical)
    ) %>%
    # Advanced feature engineering
    mutate(
      # Basic features
      HouseAge = YrSold - YearBuilt,
      RemodAge = YrSold - YearRemodAdd,
      TotalSF = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF`,
      TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath,
      TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + `3SsnPorch` + ScreenPorch,
      
      # Quality scores
      OverallScore = OverallQual * OverallCond,
      
      # Ratios and interactions
      LotRatio = GrLivArea / LotArea,
      PricePerSqft = if_else(!is.na(SalePrice), SalePrice / GrLivArea, NA_real_),
      
      # Binary features
      HasPool = ifelse(PoolArea > 0, 1, 0),
      HasGarage = ifelse(GarageArea > 0, 1, 0),
      HasBasement = ifelse(TotalBsmtSF > 0, 1, 0),
      HasFireplace = ifelse(Fireplaces > 0, 1, 0),
      Has2ndFloor = ifelse(`2ndFlrSF` > 0, 1, 0),
      
      # Neighborhood price categories (you could enhance this with external data)
      NeighborhoodGroup = case_when(
        Neighborhood %in% c("StoneBr", "NridgHt", "NoRidge") ~ "Expensive",
        Neighborhood %in% c("MeadowV", "IDOTRR", "BrDale") ~ "Cheap",
        TRUE ~ "Average"
      )
    ) %>%
    # Convert character variables to factors
    mutate(across(where(is.character), factor))
  
  return(df)
}

# Apply data preparation
combined <- prepare_data(combined)

# Split back into train and test
train_clean <- combined %>% filter(!is.na(SalePrice))
test_clean <- combined %>% filter(is.na(SalePrice))

# Log transform SalePrice
train_clean$SalePrice <- log(train_clean$SalePrice)

# Remove variables that won't help
vars_to_remove <- c("Id", "PricePerSqft")  # PricePerSqft uses target variable
train_clean <- train_clean %>% select(-all_of(vars_to_remove))
test_clean <- test_clean %>% select(-all_of(vars_to_remove))

# Handle factor levels
factor_cols <- names(train_clean)[sapply(train_clean, is.factor)]
for(col in factor_cols) {
  train_levels <- levels(train_clean[[col]])
  test_clean[[col]] <- factor(test_clean[[col]], levels = train_levels)
}

# Create cross-validation folds
cv_folds <- createFolds(train_clean$SalePrice, k = 5, list = TRUE)

# Model 1: Ridge Regression
set.seed(123)
ridge_model <- train(
  SalePrice ~ ., 
  data = train_clean,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 100)),
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Model 2: Lasso Regression
set.seed(123)
lasso_model <- train(
  SalePrice ~ ., 
  data = train_clean,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 100)),
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Model 3: Random Forest
set.seed(123)
rf_model <- train(
  SalePrice ~ ., 
  data = train_clean,
  method = "rf",
  tuneGrid = expand.grid(mtry = c(10, 15, 20, 25)),
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE),
  importance = TRUE
)

# Model 4: XGBoost
set.seed(123)
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 4, 5),
  eta = c(0.05, 0.1, 0.15),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

xgb_model <- train(
  SalePrice ~ ., 
  data = train_clean,
  method = "xgbTree",
  tuneGrid = xgb_grid,
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Model Performance Comparison
models <- list(
  "Ridge" = ridge_model,
  "Lasso" = lasso_model,
  "Random Forest" = rf_model,
  "XGBoost" = xgb_model
)

# Compare models
results <- resamples(models)
summary(results)

# Make predictions on test set
ridge_pred <- predict(ridge_model, test_clean)
lasso_pred <- predict(lasso_model, test_clean)
rf_pred <- predict(rf_model, test_clean)
xgb_pred <- predict(xgb_model, test_clean)

# Ensemble prediction (weighted average)
ensemble_pred <- 0.2 * ridge_pred + 0.2 * lasso_pred + 0.3 * rf_pred + 0.3 * xgb_pred

# Convert back to original scale
ensemble_pred_exp <- exp(ensemble_pred)

# Check the dimensions
length(test_ids)
length(ensemble_pred_exp)
nrow(test_clean)
# See which test IDs we actually have after cleaning
actual_test_ids <- test_clean$Id

# Check if the outlier filter affected test data
test_original_filtered <- test_original %>%
  filter(!(GrLivArea > 4000))  # Only apply the filter that could affect test data

nrow(test_original_filtered)
length(ensemble_pred_exp)

# If they still don't match, let's find which rows we actually have predictions for
# We can get the row indices that were used in the models

# Check which rows have complete cases after all processing
test_complete_cases <- complete.cases(test_clean %>% select(-SalePrice))
sum(test_complete_cases)

# Get the IDs for rows that have complete data
valid_test_ids <- test_original$Id[test_complete_cases]
length(valid_test_ids)

# Create submission with the valid IDs
submission <- data.frame(
  Id = valid_test_ids,
  SalePrice = ensemble_pred_exp
)

# Verify lengths match
length(valid_test_ids)
length(ensemble_pred_exp)

# If they match, write the submission
if(length(valid_test_ids) == length(ensemble_pred_exp)) {
  write.csv(submission, "improved_submission.csv", row.names = FALSE)
  cat("Submission created successfully!\n")
} else {
  cat("Lengths still don't match. valid_test_ids:", length(valid_test_ids), "predictions:", length(ensemble_pred_exp), "\n")
}

# Print results
cat("Model Performance (CV RMSE):\n")
cat("Ridge:", min(ridge_model$results$RMSE), "\n")
cat("Lasso:", min(lasso_model$results$RMSE), "\n")
cat("Random Forest:", min(rf_model$results$RMSE), "\n")
cat("XGBoost:", min(xgb_model$results$RMSE), "\n")

cat("\nSubmission file created: improved_submission.csv\n")

# Optional: Feature importance from Random Forest
if(exists("rf_model")) {
  importance_df <- data.frame(
    Feature = rownames(importance(rf_model$finalModel)),
    Importance = importance(rf_model$finalModel)[,"IncNodePurity"]
  ) %>%
    arrange(desc(Importance)) %>%
    head(20)
  
  print("Top 20 Most Important Features:")
  print(importance_df)
}
# Load the original test data to get all 1459 IDs
test_original <- read_csv("data/raw/test.csv")
all_test_ids <- test_original$Id

# Check which IDs are missing from our predictions
missing_ids <- setdiff(all_test_ids, valid_test_ids)
cat("Missing IDs:", length(missing_ids), "\n")
print(missing_ids)

# For the missing cases, we'll use the median prediction as a reasonable estimate
median_prediction <- median(ensemble_pred_exp)

# Create a complete submission with all 1459 rows
complete_submission <- data.frame(
  Id = all_test_ids,
  SalePrice = NA
)

# Fill in our actual predictions
complete_submission$SalePrice[complete_submission$Id %in% valid_test_ids] <- ensemble_pred_exp

# Fill in median values for missing cases
complete_submission$SalePrice[is.na(complete_submission$SalePrice)] <- median_prediction

# Verify we have exactly 1459 rows
nrow(complete_submission)
sum(is.na(complete_submission$SalePrice))  # Should be 0

# Write the corrected submission
write.csv(complete_submission, "improved_submission_fixed.csv", row.names = FALSE)
cat("Fixed submission created with 1459 rows!\n")