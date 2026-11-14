# Refined Reweighing (RRW-style) on Adult, German Credit, COMPAS
# Generalized adaptation, not the original LP optimization.

# 1. Packages ---------------------------------------------------------------
required_packages <- c("pROC", "FNN")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(pROC)
  library(FNN)
})
set.seed(77)

# 2. Dataset loading --------------------------------------------------------
load_adult <- function() {
  base <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
  cols <- c("age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income")
  tr <- read.csv(paste0(base, "adult.data"), header = FALSE,
                 strip.white = TRUE, na.strings = "?",
                 stringsAsFactors = FALSE)
  te <- read.csv(paste0(base, "adult.test"), header = FALSE, skip = 1,
                 strip.white = TRUE, na.strings = "?",
                 stringsAsFactors = FALSE)
  names(tr) <- cols
  names(te) <- cols
  d <- rbind(tr, te)
  d$income <- gsub("\\.", "", trimws(d$income))

  keep_cols <- c("age", "workclass", "education", "education_num",
                 "marital_status", "occupation", "relationship", "race",
                 "sex", "capital_gain", "capital_loss", "hours_per_week",
                 "income")
  d <- d[complete.cases(d[, keep_cols]), ]

  d$y <- as.integer(d$income == ">50K")  # favourable: income > 50K
  d$protected_group <- ifelse(d$race == "White" & d$sex == "Male",
                              "Privileged", "Unprivileged")
  d$age_refine <- cut(d$age, breaks = c(0, 25, 35, 45, 55, 65, Inf),
                      right = FALSE, include.lowest = TRUE)
  d$education_refine <- d$education

  features <- c("age", "workclass", "education_num", "marital_status",
                "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week")
  d <- d[, c("y", "protected_group", "age_refine", "education_refine",
             features)]
  list(data = d, name = "Adult",
       refine = c("age_refine", "education_refine"),
       privileged = "Privileged")
}

load_german <- function() {
  url <- paste0("https://archive.ics.uci.edu/ml/machine-learning-databases/",
                "statlog/german/german.data")
  cols <- c("checking_status", "duration", "credit_history", "purpose",
            "credit_amount", "savings", "employment", "installment_rate",
            "personal_status", "other_debtors", "residence_since",
            "property", "age", "other_installment_plans", "housing",
            "existing_credits", "job", "dependents", "telephone",
            "foreign_worker", "credit_label")
  d <- read.table(url, header = FALSE, col.names = cols,
                  stringsAsFactors = FALSE)
  # A92/A95 = female categories; others male.
  gender <- ifelse(d$personal_status %in% c("A92", "A95"), "Female", "Male")
  d$y <- as.integer(d$credit_label == 1)  # 1 = good credit
  d$protected_group <- ifelse(gender == "Male", "Privileged", "Unprivileged")
  d$age_refine <- cut(d$age, breaks = c(0, 25, 35, 45, 55, 65, Inf),
                      right = FALSE, include.lowest = TRUE)
  d$status_refine <- d$personal_status

  features <- c("checking_status", "duration", "credit_history", "purpose",
                "credit_amount", "savings", "employment", "installment_rate",
                "personal_status", "other_debtors", "residence_since",
                "property", "age", "other_installment_plans", "housing",
                "existing_credits", "job", "dependents", "telephone",
                "foreign_worker")
  d <- d[, c("y", "protected_group", "age_refine", "status_refine",
             features)]
  list(data = d, name = "German Credit",
       refine = c("age_refine", "status_refine"),
       privileged = "Privileged")
}

load_compas <- function() {
  url <- paste0("https://raw.githubusercontent.com/propublica/",
                "compas-analysis/master/compas-scores-two-years.csv")
  d <- read.csv(url, stringsAsFactors = FALSE)
  d <- d[d$days_b_screening_arrest <= 30 &
           d$days_b_screening_arrest >= -30 &
           d$is_recid != -1 &
           d$c_charge_degree != "O" &
           !is.na(d$score_text), ]

  d$y <- as.integer(d$two_year_recid == 0)  # favourable: no recidivism
  d$protected_group <- ifelse(d$race == "Caucasian" & d$sex == "Male",
                              "Privileged", "Unprivileged")
  d$age_refine <- cut(d$age, breaks = c(0, 25, 35, 45, 55, 65, Inf),
                      right = FALSE, include.lowest = TRUE)
  d$charge_refine <- d$c_charge_degree

  features <- c("age", "race", "sex", "c_charge_degree", "priors_count",
                "juv_fel_count", "juv_misd_count", "juv_other_count")
  d <- d[, c("y", "protected_group", "age_refine", "charge_refine",
             features)]
  d <- d[complete.cases(d), ]
  list(data = d, name = "COMPAS",
       refine = c("age_refine", "charge_refine"),
       privileged = "Privileged")
}

# 3. Fairness metrics -------------------------------------------------------
selection_rates <- function(yhat, group, privileged_label) {
  group <- as.character(group)
  c(privileged   = mean(yhat[group == privileged_label]),
    unprivileged = mean(yhat[group != privileged_label]))
}

# Unprivileged minus privileged positive rate (negative = disadvantage).
dp_difference <- function(yhat, group, privileged_label) {
  r <- selection_rates(yhat, group, privileged_label)
  unname(r["unprivileged"] - r["privileged"])
}

disparate_impact <- function(yhat, group, privileged_label) {
  r <- selection_rates(yhat, group, privileged_label)
  if (is.na(r["privileged"]) || r["privileged"] == 0) return(NA_real_)
  unname(r["unprivileged"] / r["privileged"])
}

# kNN consistency on a reproducible sample.
calculate_consistency <- function(x, yhat, k = 5, max_n = 2000) {
  n <- nrow(x)
  if (n <= 2) return(NA_real_)
  set.seed(77)
  idx <- if (n > max_n) sample(seq_len(n), max_n) else seq_len(n)
  xx <- as.matrix(x[idx, , drop = FALSE])
  yy <- yhat[idx]
  storage.mode(xx) <- "double"
  xx[!is.finite(xx)] <- 0

  keep <- apply(xx, 2, function(z) length(unique(z)) > 1)
  xx <- xx[, keep, drop = FALSE]
  if (ncol(xx) == 0) return(NA_real_)
  xx <- scale(xx)
  xx[!is.finite(xx)] <- 0

  nn <- FNN::get.knnx(data = xx, query = xx,
                      k = min(k + 1, nrow(xx)))$nn.index
  if (ncol(nn) < 2) return(NA_real_)

  agree <- numeric(nrow(xx))
  for (i in seq_len(nrow(xx))) {
    nb <- nn[i, ]
    nb <- head(nb[nb != i], k)
    agree[i] <- if (length(nb) == 0) NA_real_ else mean(yy[i] == yy[nb])
  }
  mean(agree, na.rm = TRUE)
}

evaluate_predictions <- function(data, yhat, prob, x_for_consistency,
                                 privileged_label) {
  cm <- table(factor(yhat, levels = c(0, 1)),
              factor(data$y, levels = c(0, 1)))
  fn <- cm[1, 2]; fp <- cm[2, 1]; tp <- cm[2, 2]
  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  recall    <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0)
    NA_real_ else 2 * precision * recall / (precision + recall)

  auc_value <- tryCatch(
    as.numeric(pROC::auc(pROC::roc(response = data$y, predictor = prob,
                                   levels = c(0, 1), direction = "<",
                                   quiet = TRUE))),
    error = function(e) NA_real_
  )

  data.frame(
    DP_Diff     = dp_difference(yhat, data$protected_group, privileged_label),
    DIR         = disparate_impact(yhat, data$protected_group,
                                   privileged_label),
    Consistency = calculate_consistency(x_for_consistency, yhat),
    Accuracy    = mean(yhat == data$y),
    Precision   = precision,
    Recall      = recall,
    F1          = f1,
    AUC         = auc_value
  )
}

# 4. Stratified split -------------------------------------------------------
stratified_split <- function(d, p = 0.80) {
  strata <- interaction(d$y, d$protected_group, drop = TRUE)
  train_idx <- unlist(lapply(split(seq_len(nrow(d)), strata), function(ix) {
    ix[sample.int(length(ix), size = max(1, round(p * length(ix))))]
  }))
  list(train = d[sort(train_idx), , drop = FALSE],
       test  = d[-train_idx, , drop = FALSE])
}

# 5. Refined reweighing weights ---------------------------------------------
# weight = P(A) * P(Y, X-bin) / P(A, Y, X-bin), i.e. P(A) / P(A | Y, X-bin),
# with additive smoothing. Capped at the 99th percentile, normalized to mean 1.
make_refinement_bins <- function(data, refine_vars) {
  out <- data.frame(row.names = seq_len(nrow(data)))
  for (v in refine_vars) {
    z <- data[[v]]
    if (is.numeric(z)) {
      brks <- unique(quantile(z, probs = seq(0, 1, 0.2), na.rm = TRUE))
      out[[v]] <- as.character(cut(z, breaks = brks, include.lowest = TRUE))
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
  x_key <- do.call(paste, c(bins, sep = " || "))

  key_xy  <- paste(y, x_key, sep = " || ")
  key_axy <- paste(a, y, x_key, sep = " || ")

  n_a   <- as.numeric(table(a)[a])
  n_y   <- as.numeric(table(y)[y])
  n_xy  <- as.numeric(table(key_xy)[key_xy])
  n_axy <- as.numeric(table(key_axy)[key_axy])

  k_a   <- length(unique(a))
  k_y   <- length(unique(y))
  k_xy  <- length(unique(key_xy))
  k_axy <- length(unique(key_axy))

  pa  <- (n_a + smoothing) / (n + smoothing * k_a)
  pxy <- (n_xy + smoothing) / (n + smoothing * k_xy)
  paxy <- (n_axy + smoothing) / (n + smoothing * k_axy)

  w <- (pa * pxy) / paxy
  cap <- as.numeric(quantile(w, 0.99, na.rm = TRUE))
  w <- pmin(w, cap)
  w / mean(w)
}

# 6. Model fitting and experiment runner ------------------------------------
fit_logistic <- function(train, test, weights = NULL) {
  predictors <- setdiff(names(train),
                        c("y", "protected_group", "age_refine",
                          "education_refine", "status_refine",
                          "charge_refine"))
  fml <- reformulate(predictors, response = "y")

  # Make sure train and test share factor levels.
  combined <- rbind(train[, c("y", predictors), drop = FALSE],
                    test[,  c("y", predictors), drop = FALSE])
  for (v in predictors) {
    if (is.character(combined[[v]]) || is.factor(combined[[v]])) {
      combined[[v]] <- factor(combined[[v]])
    }
  }
  train2 <- combined[seq_len(nrow(train)), , drop = FALSE]
  test2  <- combined[nrow(train) + seq_len(nrow(test)), , drop = FALSE]

  model <- suppressWarnings(
    if (is.null(weights)) {
      glm(fml, data = train2, family = binomial())
    } else {
      glm(fml, data = train2, family = quasibinomial(), weights = weights)
    }
  )

  prob <- suppressWarnings(predict(model, newdata = test2, type = "response"))
  prob[!is.finite(prob)] <- mean(train$y)
  pred <- as.integer(prob >= 0.5)
  x <- model.matrix(fml, data = test2)
  list(model = model, probability = prob, prediction = pred, x = x)
}

run_experiment <- function(dataset) {
  cat("\n========================================\n")
  cat("Running:", dataset$name, "\n")
  cat("========================================\n")

  d <- dataset$data
  d$protected_group <- factor(d$protected_group)
  d$y <- as.integer(d$y)

  sp <- stratified_split(d, p = 0.80)
  train <- sp$train
  test  <- sp$test

  baseline <- fit_logistic(train, test, weights = NULL)
  baseline_metrics <- evaluate_predictions(
    test, baseline$prediction, baseline$probability, baseline$x,
    dataset$privileged)

  # Weights estimated on the training split only.
  weights <- rrw_weights(train, dataset$refine)
  weighted <- fit_logistic(train, test, weights = weights)
  weighted_metrics <- evaluate_predictions(
    test, weighted$prediction, weighted$probability, weighted$x,
    dataset$privileged)

  original_label_metrics <- data.frame(
    DP_Diff = dp_difference(train$y, train$protected_group,
                            dataset$privileged),
    DIR = disparate_impact(train$y, train$protected_group,
                           dataset$privileged),
    Consistency = NA_real_
  )

  result <- rbind(
    cbind(Dataset = dataset$name, Stage = "Original labels (train)",
          original_label_metrics,
          Accuracy = NA_real_, Precision = NA_real_, Recall = NA_real_,
          F1 = NA_real_, AUC = NA_real_),
    cbind(Dataset = dataset$name, Stage = "Baseline model (test)",
          baseline_metrics),
    cbind(Dataset = dataset$name, Stage = "RRW model (test)",
          weighted_metrics)
  )

  list(results = result, weights = weights,
       baseline_model = baseline$model, weighted_model = weighted$model)
}

# 7. Run all datasets -------------------------------------------------------
datasets <- list(load_adult(), load_german(), load_compas())
experiment_outputs <- lapply(datasets, run_experiment)
results <- do.call(rbind, lapply(experiment_outputs, `[[`, "results"))

# 8. Display and save results -----------------------------------------------
num_cols <- vapply(results, is.numeric, logical(1))
results[num_cols] <- lapply(results[num_cols], round, 4)
print(results, row.names = FALSE)

write.csv(results, "fairness_results.csv", row.names = FALSE)
saveRDS(experiment_outputs, "fairness_model_objects.rds")
cat("\nFinished. Saved:\n - fairness_results.csv\n - fairness_model_objects.rds\n")
