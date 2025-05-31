# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(randomForest)
library(xgboost)
library(caret)


# Set your working directory 
setwd("~/GitHub-House-Prices-Regression-Final-Project/House_Prices_Regression")

# Load the datasets
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# View the first few rows
head(train)

# View the structure of the data
glimpse(train)

# Summary statistics
summary(train)

# Look at the distribution of SalePrice
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(fill = "skyblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Sale Price")

# Count missing values by column
colSums(is.na(train))
