
install.packages(c(
  "tidyverse",
  "caret",
  "lpSolve",
  "pROC",
  "FNN",
  "yardstick"
))

library(tidyverse)
library(caret)
library(lpSolve)
library(pROC)
library(FNN)
library(yardstick)



# Demographic Parity Difference
dp_difference <- function(y, protected,
                          unprivileged, privileged) {

  rate_unpriv <- mean(y[protected == unprivileged] == 1)
  rate_priv <- mean(y[protected == privileged] == 1)

  rate_unpriv - rate_priv
}


# Disparate Impact Ratio
disparate_impact <- function(y, protected,
                             unprivileged, privileged) {

  rate_unpriv <- mean(y[protected == unprivileged] == 1)
  rate_priv <- mean(y[protected == privileged] == 1)

  if (rate_priv == 0) return(NA_real_)

  rate_unpriv / rate_priv
}


# Consistency
calculate_consistency <- function(X, y, k = 5) {

  X <- as.data.frame(X)

  # Convert categorical variables into numeric indicators
  X <- model.matrix(~ . - 1, data = X)
  
  X <- X[, apply(X, 2, sd) > 0, drop = FALSE]

  # Standardizing
  X <- scale(X)

  # Find KNN
  neighbors <- FNN::get.knnx(
    data = X,
    query = X,
    k = k + 1
  )$nn.index[, -1, drop = FALSE]

  neighbor_labels <- matrix(
    y[as.vector(neighbors)],
    nrow = nrow(neighbors)
  )

  1 - mean(abs(
    matrix(y, nrow = length(y), ncol = k) -
      neighbor_labels
  ))
}

original_metrics <- function(data, outcome, protected,
                             unprivileged, privileged,
                             features) {

  y <- data[[outcome]]

  tibble(
    DP_Diff = dp_difference(
      y, data[[protected]], unprivileged, privileged
    ),
    DIR = disparate_impact(
      y, data[[protected]], unprivileged, privileged
    ),
    Consistency = calculate_consistency(
      data[, features, drop = FALSE], y
    )
  )
}

split_data <- function(data, outcome, seed = 123) {

  set.seed(seed)

  train_index <- caret::createDataPartition(
    data[[outcome]],
    p = 0.80,
    list = FALSE
  )

  list(
    train = data[train_index, ],
    test = data[-train_index, ]
  )
}

train_model <- function(train, test, outcome,
                        predictors, weights = NULL) {

  formula <- as.formula(
    paste(outcome, "~", paste(predictors, collapse = " + "))
  )

  if (is.null(weights)) {
    model <- glm(
      formula,
      data = train,
      family = binomial()
    )
  } else {
    model <- glm(
      formula,
      data = train,
      family = binomial(),
      weights = weights
    )
  }

  probabilities <- predict(
    model,
    newdata = test,
    type = "response"
  )

  predictions <- ifelse(probabilities >= 0.5, 1, 0)

  list(
    model = model,
    probabilities = probabilities,
    predictions = predictions
  )
}

evaluate_model <- function(test, predictions,
                           outcome, protected,
                           unprivileged, privileged,
                           features) {

  actual <- test[[outcome]]

  tibble(
    DP_Diff = dp_difference(
      predictions,
      test[[protected]],
      unprivileged,
      privileged
    ),

    DIR = disparate_impact(
      predictions,
      test[[protected]],
      unprivileged,
      privileged
    ),

    Consistency = calculate_consistency(
      test[, features, drop = FALSE],
      predictions
    ),

    Accuracy = mean(predictions == actual),

    Precision = yardstick::precision_vec(
      factor(actual, levels = c(0, 1)),
      factor(predictions, levels = c(0, 1))
    ),

    Recall = yardstick::recall_vec(
      factor(actual, levels = c(0, 1)),
      factor(predictions, levels = c(0, 1))
    ),

    F1 = yardstick::f_meas_vec(
      factor(actual, levels = c(0, 1)),
      factor(predictions, levels = c(0, 1))
    ),

    AUC = as.numeric(
      pROC::auc(actual, predictions)
    )
  )
}



run_experiment <- function(data, outcome, protected,
                           unprivileged, privileged,
                           predictors, features,
                           rrw_function) {

  # 1. Original dataset metrics
  original <- original_metrics(
    data, outcome, protected,
    unprivileged, privileged, features
  )

  # 2. Stratified split
  split <- split_data(data, outcome)

  train <- split$train
  test <- split$test

  # 3. Baseline model (no mitigation)
  baseline <- train_model(
    train, test, outcome, predictors
  )

  baseline_results <- evaluate_model(
    test,
    baseline$predictions,
    outcome, protected,
    unprivileged, privileged, features
  )

  # 4. Apply RRW to training data only
  rrw_weights <- rrw_function(train)

  # 5. Train weighted model
  mitigated <- train_model(
    train, test, outcome, predictors,
    weights = rrw_weights
  )

  # 6. Evaluate mitigated model
  mitigated_results <- evaluate_model(
    test,
    mitigated$predictions,
    outcome, protected,
    unprivileged, privileged, features
  )

  # 7. Combine results
  results <- bind_rows(
    original %>% mutate(Stage = "Original labels"),
    baseline_results %>% mutate(Stage = "Baseline predictions"),
    mitigated_results %>% mutate(Stage = "RRW predictions")
  )

  results %>%
    select(Stage, everything())
}


