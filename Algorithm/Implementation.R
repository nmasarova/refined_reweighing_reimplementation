############################################################
# IMPLEMENTATION.R
# Comparative fairness experiment: Adult, German Credit, COMPAS
#
# IMPORTANT METHODOLOGICAL NOTE:
# This is a runnable, generalized two-stage reweighing
# implementation for comparing the three datasets.
# It is NOT a verbatim reproduction of the original
# Adult-specific Phase 1 + Phase 2 LP algorithm in
# phases_1_2_adult.R. That original code is tied to a
# particular 8-cell race x gender x income setup and its
# age/education refinement. The generalized procedure
# below reweights protected-group x outcome x covariate-bin
# cells, so results must be described as a generalized
# RRW-style adaptation, not as exact reproduction.
#
# Outcome convention:
#   y = 1 means favourable outcome.
# Fairness convention:
#   DP_Diff = P(predicted positive | unprivileged)
#             - P(predicted positive | privileged)
#   DIR = unprivileged selection rate / privileged rate
#
# Run this whole file from RStudio or with:
#   Rscript Implementation.R
############################################################

############################
# 1. Packages
############################

required_packages <- c("caret", "pROC", "FNN")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(caret)
  library(pROC)
  library(FNN)
})

set.seed(77)

############################
# 2. Dataset loading
############################

# All data are downloaded from public sources.
# If download fails, download the files manually and change
# the URL/path in the relevant loader.

load_adult <- function() {
  train_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  test_url  <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

  cols <- c(
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
  )

  tr <- read.csv(train_url, header = FALSE, strip.white = TRUE,
                 na.strings = "?", stringsAsFactors = FALSE)
  te <- read.csv(test_url, header = FALSE, skip = 1, strip.white = TRUE,
                 na.strings = "?", stringsAsFactors = FALSE)
  names(tr) <- cols
  names(te) <- cols

  d <- rbind(tr, te)
  d$income <- gsub("\\.", "", trimws(d$income))
  d <- d[complete.cases(d[, c("age", "education", "race", "sex", "income",
                              "workclass", "marital_status", "occupation",
                              "relationship", "hours_per_week")]), ]

  # Favourable outcome: earning more than $50K.
  d$y <- as.integer(d$income == ">50K")
  d$protected_group <- ifelse(d$race == "White" & d$sex == "Male",
                              "Privileged", "Unprivileged")

        refine = c("age_refine", "education_refine"),
       privileged = "Privileged")
}

load_german <- function() {
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
  cols <- c("checking_status", "duration", "credit_history", "purpose",
            "credit_amount", "savings", "employment", "installment_rate",
            "personal_status", "other_debtors", "residence_since",
            "property", "age", "other_installment_plans", "housing",
            "existing_credits", "job", "dependents", "telephone",
            "foreign_worker", "credit_label")
  d <- read.table(url, header = FALSE, col.names = cols,
                  stringsAsFactors = FALSE)
  # A91/A93/A94 are male categories; A92/A95 are female categories.
  d$gender <- ifelse(d$personal_status %in% c("A92", "A95"),
                     "Female", "Male")
  # German data: 1 = good credit, 2 = bad credit.
  d$y <- as.integer(d$credit_label == 1)
  d$protected_group <- ifelse(d$gender == "Male",
                              "Privileged", "Unprivileged")

  d$age_refine <- cut(d$age,
                      breaks = c(0, 25, 35, 45, 55, 65, Inf),
                      right = FALSE, include.lowest = TRUE)
  d$status_refine <- d$personal_status

  features <- c("checking_status", "duration", "credit_history",
                "purpose", "credit_amount", "savings", "employment",
                "installment_rate", "personal_status", "other_debtors",
                "residence_since", "property", "age",
                "other_installment_plans", "housing", "existing_credits",
                "job", "dependents", "telephone", "foreign_worker", "gender")
  d <- d[, c("y", "protected_group", "age_refine",
             "status_refine", features)]
  list(data = d, name = "German Credit",
       protected = "protected_group",
       refine = c("age_refine", "status_refine"),
       privileged = "Privileged")
}

load_compas <- function() {
  url <- paste0(
    "https://raw.githubusercontent.com/propublica/compas-analysis/",
    "master/compas-scores-two-years.csv"
  )
  d <- read.csv(url, stringsAsFactors = FALSE)

  needed <- c("race", "sex", "age", "c_charge_degree",
              "two_year_recid", "days_b_screening_arrest",
              "is_recid", "c_charge_desc", "priors_count",
              "juv_fel_count", "juv_misd_count", "juv_other_count")
  d <- d[d$days_b_screening_arrest <= 30 &
           d$days_b_screening_arrest >= -30 &
           d$is_recid != -1 &
           d$c_charge_degree != "O" &
           !is.na(d$score_text), ]

  # Favourable outcome: no recidivism within two years.
  d$y <- as.integer(d$two_year_recid == 0)
  d$protected_group <- ifelse(d$race == "Caucasian" & d$sex == "Male",
                              "Privileged", "Unprivileged")
  d$age_refine <- cut(d$age,
                      breaks = c(0, 25, 35, 45, 55, 65, Inf),
                      right = FALSE, include.lowest = TRUE)
  d$charge_refine <- d$c_charge_degree

  features <- c("age", "race", "sex", "c_charge_degree",
                "priors_count", "juv_fel_count", "juv_misd_count",
                "juv_other_count")
  d <- d[, c("y", "protected_group", "age_refine",
             "charge_refine", features)]
  d <- d[complete.cases(d), ]
  list(data = d, name = "COMPAS",
       protected = "protected_group",
       refine = c("age_refine", "charge_refine"),
       privileged = "Privileged")
}

}

disparate_impact <- function(yhat, group, privileged_label) {
  rates <- selection_rates(yhat, group, privileged_label)
  if (is.na(rates["privileged"]) || rates["privileged"] == 0) return(NA_real_)
  unname(rates["unprivileged"] / rates["privileged"])
}

# k-nearest-neighbour consistency, estimated on a reproducible
# sample to keep computation practical for Adult.
calculate_consistency <- function(x, yhat, k = 5, max_n = 2000) {
  n <- nrow(x)
  if (n <= 2) return(NA_real_)
  set.seed(77)
  idx <- if (n > max_n) sample(seq_len(n), max_n) else seq_len(n)
  xx <- x[idx, , drop = FALSE]
  yy <- yhat[idx]
  xx <- as.matrix(xx)
  storage.mode(xx) <- "double"
  xx[!is.finite(xx)] <- 0

  # Remove constant columns; scale numeric design matrix.
  keep <- apply(xx, 2, function(z) length(unique(z)) > 1)
  xx <- xx[, keep, drop = FALSE]
  if (ncol(xx) == 0) return(NA_real_)
  xx <- scale(xx)
  xx[!is.finite(xx)] <- 0

  nn <- FNN::get.knnx(data = xx, query = xx, k = min(k + 1, nrow(xx)))$nn.index
  if (ncol(nn) < 2) return(NA_real_)

  neighbour_agreement <- numeric(nrow(xx))
  for (i in seq_len(nrow(xx))) {
    neighbours <- nn[i, ]
    neighbours <- neighbours[neighbours != i]
    neighbours <- head(neighbours, k)
    if (length(neighbours) == 0) {
      neighbour_agreement[i] <- NA_real_
    } else {
      neighbour_agreement[i] <- mean(yy[i] == yy[neighbours])
    }
  }
  mean(neighbour_agreement, na.rm = TRUE)
}

evaluate_predictions <- function(data, yhat, prob, x_for_consistency,
                                 privileged_label) {
  cm <- table(factor(yhat, levels = c(0, 1)),
              factor(data$y, levels = c(0, 1)))
  tn <- cm[1, 1]; fn <- cm[1, 2]
  fp <- cm[2, 1]; tp <- cm[2, 2]
  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  f1 <- if (is.na(precision) || is.na(recall) ||
            (precision + recall) == 0) NA_real_ else
              2 * precision * recall / (precision + recall)

  auc_value <- tryCatch(
    as.numeric(pROC::auc(response = data$y, predictor = prob,
                         quiet = TRUE)),
    error = function(e) NA_real_
  )

  data.frame(
    DP_Diff = dp_difference(yhat, data$protected_group, privileged_label),
    DIR = disparate_impact(yhat, data$protected_group, privileged_label),
    Consistency = calculate_consistency(x_for_consistency, yhat),
    Accuracy = mean(yhat == data$y),
    Precision = precision,
    Recall = recall,
    F1 = f1,
    AUC = auc_value
  )
}

# Weights are normalized to have mean 1.
#
# This is a generalized RRW-style adaptation. It is not the
# original LP optimization from the Adult-specific script.

make_refinement_bins <- function(data, refine_vars) {
  out <- data.frame(row.names = seq_len(nrow(data)))
  for (v in refine_vars) {
    z <- data[[v]]
    if (is.numeric(z)) {
      out[[v]] <- cut(z,
                      breaks = unique(quantile(z,
                                               probs = seq(0, 1, 0.2),
                                               na.rm = TRUE)),
                      include.lowest = TRUE)
    } else {
      out[[v]] <- as.character(z)
    }
    out[[v]][is.na(out[[v]])] <- "Missing"
  }
  out
}

rrw_weights <- function(train, refine_vars, smoothing = 0.5) {
  n <- nrow(train)
  if (n == 0) stop("Training data has zero rows.")

  a <- as.character(train$protected_group)
  y <- as.character(train$y)
  bins <- make_refinement_bins(train, refine_vars)

  # Combined refinement stratum.
  x_key <- do.call(paste, c(bins, sep = " || "))
  a_levels <- unique(a)
  y_levels <- unique(y)
  x_levels <- unique(x_key)

  # Smoothed marginal counts.
  count_a <- table(factor(a, levels = a_levels))
  count_y <- table(factor(y, levels = y_levels))
  count_xy <- table(factor(paste(y, x_key, sep = " || ")))
  count_axy <- table(factor(paste(a, y, x_key, sep = " || ")))

  key_axy <- paste(a, y, x_key, sep = " || ")
  key_xy <- paste(y, x_key, sep = " || ")

  # P(A) * P(Y) * P(X-bin | Y) / P(A,Y,X-bin)
  # with additive smoothing in the relevant contingency tables.
  weights <- numeric(n)
  for (i in seq_len(n)) {
    na <- as.numeric(count_a[a[i]]) + smoothing
    ny <- as.numeric(count_y[y[i]]) + smoothing
    nxy <- as.numeric(count_xy[key_xy[i]]) + smoothing
    naxy <- as.numeric(count_axy[key_axy[i]]) + smoothing

    pa <- na / (n + smoothing * length(a_levels))
    py <- ny / (n + smoothing * length(y_levels))
    pxy_given_y <- nxy / (ny + smoothing * length(x_levels))
    paxy <- naxy / (n + smoothing * length(unique(key_axy)))

    weights[i] <- (pa * py * pxy_given_y) / paxy
  }

  # Stabilize extreme weights and normalize.
  cap <- as.numeric(quantile(weights, 0.99, na.rm = TRUE))
  weights <- pmin(weights, cap)
  weights <- weights / mean(weights)
  weights
}


############################
# 6. Model fitting and experiment runner
############################

fit_logistic <- function(train, test, weights = NULL) {
  predictors <- setdiff(names(train),
                        c("y", "age_refine", "education_refine",
                          "status_refine", "charge_refine"))
  formula <- reformulate(predictors, response = "y")

  # Ensure train and test factor levels match.
  combined <- rbind(train[, c("y", predictors), drop = FALSE],
                    test[, c("y", predictors), drop = FALSE])
  for (v in predictors) {
    if (is.character(combined[[v]]) || is.factor(combined[[v]])) {
      combined[[v]] <- factor(combined[[v]])
    }
  }
  train2 <- combined[seq_len(nrow(train)), , drop = FALSE]
  test2 <- combined[nrow(train) + seq_len(nrow(test)), , drop = FALSE]

  model_data <- train2
  if (is.null(weights)) {
    model <- glm(formula, data = model_data, family = binomial())
  } else {
    model <- glm(formula, data = model_data, family = binomial(),
                 weights = weights)
  }

  prob <- predict(model, newdata = test2, type = "response")
  prob[!is.finite(prob)] <- mean(train$y)
  pred <- as.integer(prob >= 0.5)

  # Consistency uses the predictor matrix for the held-out rows.
  x <- model.matrix(formula, data = test2)
  list(model = model, probability = prob, prediction = pred,
       x = x)
}

run_experiment <- function(dataset) {
  cat("\n========================================\n")
  cat("Running:", dataset$name, "\n")
  cat("========================================\n")

  d <- dataset$data
  d$protected_group <- factor(d$protected_group)
  d$y <- as.integer(d$y)

  split <- stratified_split(d, p = 0.80)
  train <- split$train
  test <- split$test

  # Baseline (unweighted) model.
  baseline <- fit_logistic(train, test, weights = NULL)
  baseline_metrics <- evaluate_predictions(
    test, baseline$prediction, baseline$probability,
    baseline$x, dataset$privileged
  )

  # Refined reweighing weights are estimated ONLY on training data.
  weights <- rrw_weights(train, dataset$refine)
  weighted <- fit_logistic(train, test, weights = weights)
  weighted_metrics <- evaluate_predictions(
    test, weighted$prediction, weighted$probability,
    weighted$x, dataset$privileged
  )

  # Original-label fairness on the training split, before mitigation.
  original_label_metrics <- data.frame(
    DP_Diff = dp_difference(train$y, train$protected_group,
                            dataset$privileged),
    DIR = disparate_impact(train$y, train$protected_group,
                           dataset$privileged),
    Consistency = NA_real_
  )

  result <- rbind(
    cbind(Dataset = dataset$name, Stage = "Original labels (train)",
          original_label_metrics),
    cbind(Dataset = dataset$name, Stage = "Baseline model (test)",
          baseline_metrics),
    cbind(Dataset = dataset$name, Stage = "RRW model (test)",
          weighted_metrics)
  )

  list(results = result, weights = weights,
       baseline_model = baseline$model,
       weighted_model = weighted$model)
}

############################
# 7. Run all datasets
############################

datasets <- list(
  load_adult(),
  load_german(),
  load_compas()
)

experiment_outputs <- lapply(datasets, run_experiment)
results <- do.call(rbind, lapply(experiment_outputs, `[[`, "results"))

############################
# 8. Display and save results
############################

print(results, row.names = FALSE)

write.csv(results, "fairness_results.csv", row.names = FALSE)
saveRDS(experiment_outputs, "fairness_model_objects.rds")

cat("\nFinished. Saved:\n")
cat(" - fairness_results.csv\n")
cat(" - fairness_model_objects.rds\n")
