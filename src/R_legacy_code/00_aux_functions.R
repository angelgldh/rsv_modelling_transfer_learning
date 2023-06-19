#
# This scripts contains all the auxiliary functions used along the rest of the project
# They are classified in the following sections:
# - Model diagnosis functions 
# - Functions to analyze sampling methods
# - Save and load ML models in a database
# - Model performance and model fitting
# - EDA functions
# - Legacy functions
# - Setting up the project
# - Train-test-split
# - Feature selection
# - Feature build-up



# Model diagnosis functions ####
load_models_from_custom_database = function(db_name){
  
  
  db_models <- dbConnect(SQLite(), dbname = db_name)
  trained_models_names = (dbGetQuery(db_models, paste0("SELECT name FROM models"))) $ name
  dbDisconnect(db_models)
  
  trained_models_all = vector(mode = "list", length = length(trained_models_names))
  names(trained_models_all) = trained_models_names
  
  
  for (ii in 1:length(trained_models_names)){
    
    model = load_model(db_name = db_name, model_name = trained_models_names[ii])
    trained_models_all[[ii]] = model
    
  }
  
  return(trained_models_all)
}


metrics_model_list = function(model_list, validation_df){
  
  output_df = data.frame(method = character(),
                         split = character(),
                         sampling = character(),
                         fine_tuned = logical(),
                         auc = numeric(),
                         f1 = numeric(),
                         sensitivity = numeric(),
                         specificity = numeric(),
                         optimal_threshold = numeric())
  
  
  for (ii in 1:length(model_list)){
    model_target = model_list[[ii]]
    name_model = names(model_list)[ii]
    
    method = strsplit(name_model, "_")[[1]][2]
    split = strsplit(name_model, "_")[[1]][4]
    if(is.na(split)){split = "8020"}
    
    sampling = strsplit(name_model, "_")[[1]][3]
    
    fine_tuned = FALSE
    
    # find the optimal threshold
    thresholds <- seq(0.1, 0.9, by = 0.01)
    metrics_list <- lapply(thresholds, function(th) performance_metrics(model = model_target, validation_data = validation_df, 
                                                                        threshold_pred = th))
    metrics_df <- bind_rows(metrics_list) %>% mutate(threshold = thresholds)
    optimal_threshold <- metrics_df[which.max(metrics_df$f1_score), "threshold"] $ threshold
    
    # Now, the metrics
    auc = test_roc(model_target, validation_df) $ auc
    
    all_metrics = metrics_df [metrics_df $ threshold == optimal_threshold,]
    sensitivity = all_metrics $ sensitivity
    specificity = all_metrics $ specificity
    f1 = all_metrics $f1_score
    precision = all_metrics $ precision
    recall = all_metrics $recall
    ppv = all_metrics $ ppv
    npv = all_metrics $ npv
    
    
    output_df <- rbind(output_df, c(method,split, sampling, fine_tuned, round(auc,3), round(f1,3), round(sensitivity,3), round(specificity,3),
                                    round(ppv,3), round(npv,3), optimal_threshold))
    
    
  }
  
  names(output_df) <- c("method", "split", "sampling", "fine_tuned", "auc", "f1", "sensitivity", "specificity", "ppv", "npv", "optimal_threshold")
  
  return(output_df)
}


obtain_auc = function(model_list_roc){
  # Computes the AUC score for a list of ROC curves.
  #
  # Inputs:
  #   model_list_roc (list): A list of ROC curves.
  #
  # Output:
  #   A list with the AUC scores for each ROC curve in the input list.
  
  lapply(model_list_roc, function(roc) { 
    auc =  roc %>% auc() 
    return(auc)})
}

performance_metrics <- function(model, validation_data, threshold_pred) {
  # Computes several performance metrics for a binary classification model using a validation dataset.
  #
  # Inputs:
  #   model (fitted model object): The binary classification model to evaluate.
  #   validation_data (data frame): The validation dataset.
  #   threshold_pred (numeric): The probability threshold for predictions.
  #
  # Output:
  #   A list with the following performance metrics:
  #   - sensitivity
  #   - specificity
  #   - f1_score
  #   - precision
  #   - recall
  #   - ppv
  #   - npv
  #
  # Example usage:
  #   performance_metrics(
  #     model = trained_model,
  #     validation_data = validation,
  #     threshold_pred = 0.6
  #   )
  
  
  actual_response = factor(ifelse(validation_data $RSV_test_result == "Positive",1,0))
  
  predictions_probs = predict(model, validation_data, type = "prob")[, "Positive"]
  
  # Define predcitions as a 2-level factor to ensure consistency
  predictions = factor(ifelse(predictions_probs >= threshold_pred,1,0), levels = c(0,1))
  
  cm = caret::confusionMatrix(data = predictions, reference = actual_response, 
                              positive = "1")
  
  sensitivity = cm $byClass["Sensitivity"]
  specificity = cm $ byClass["Specificity"]
  f1_score = cm $ byClass["F1"]
  precision = cm $byClass["Precision"]
  recall = cm $ byClass["Recall"]
  ppv = cm $ byClass["Pos Pred Value"]
  npv = cm $ byClass["Neg Pred Value"]
  
  list(sensitivity = sensitivity, specificity = specificity, f1_score = f1_score,
       precision = precision, recall = recall,
       ppv = ppv, npv = npv)
  
}

bootstrap_metrics = function(model, validation_data, threshold_pred, n_boot = 500){
  # Function to estimate the sampling distribution of various performance metrics for a given model and validation data using bootstrapping
  # Args:
  #   model ("train" "train.formula"): The predictive model that you want to evaluate, built on caret::train()
  #   validation_data ("dataframe"): A dataset containing the validation data, with the actual response variable to compare against the model's predictions
  #   threshold_pred ("numeric): A threshold value for classifying the predicted probabilities into binary class labels (0 or 1)
  #   n_boot ("numeric"): The number of bootstrap iterations (default is 500). Increasing this number may provide more accurate estimates but will also increase the computational time
  # Returns:
  #   A list containing the following bootstrap estimates for each performance metric:
  #   f1.b: Bootstrap estimates of the F1 score
  #   sensitivity.b: Bootstrap estimates of sensitivity
  #   specificity.b: Bootstrap estimates of specificity
  #   ppv.b: Bootstrap estimates of PPV (positive predictive value)
  #   npv.b: Bootstrap estimates of NPV (negative predictive value)
  #   precision.b: Bootstrap estimates of precision
  #   recall.b: Bootstrap estimates of recall
  
  f1.b = sensitivity.b = specificity.b = precision.b = recall.b = ppv.b = npv.b = numeric(length = n_boot)
  
  actual_response = factor(ifelse(validation_data $RSV_test_result == "Positive",1,0))
  # Main boostrap loop
  
  for (rep_ in(1:n_boot)){
    indices = sample(1:length(actual_response), replace = T)
    
    metrics = performance_metrics(model, validation_data, threshold_pred)
    
    f1.b[rep] = metrics $ f1_score
    sensitivity.b[rep] = metrics $ sensitivity
    specificity.b[rep] = metrics $ specificity
    ppv.b[rep] = metrics $ ppv
    npv.b[rep] = metrics $ npv
    
    precision.b[rep] = metrics $ precision
    recall.b[rep] = metrics $ recall
    
  }
  
  return(
    list(f1.b = f1.b, sensitivity.b = sensitivity.b, specificity.b = specificity.b,
         ppv.b = ppv.b, npv.b = npv.b, 
         precision.b = precision.b, recall.b = recall.b)
  )
}

test_roc <- function(model, validation_set) {
  # Computes the ROC curve for a binary classification model using a validation dataset.
  #
  # Inputs:
  #   model (fitted model object): The binary classification model to evaluate.
  #   validation_set (data frame): The validation dataset.
  #
  # Output:
  #   The ROC curve for the model predictions.
  #
  # Example usage:
  #   test_roc(
  #     model = trained_model,
  #     validation_set = validation
  #   )
  
  if (class(model)[1] == "xgb.Booster"){
    response_variable = "RSV_test_result"
    validation_train = validation_set %>% select(-(RSV_test_result))
    
    categorical_predictors = names(validation_train)[(sapply(validation_train, class)) == "factor" ]
    
    validation_df_onehot = dummies::dummy.data.frame(validation_set, names = categorical_predictors, sep = "_")
    
    X_validation <- as.matrix(validation_df_onehot[, !(colnames(validation_df_onehot) %in% response_variable)])
    y_validation <- validation_df_onehot[, (colnames(validation_df_onehot) %in% response_variable)]
    
    xgboost_validation <- xgb.DMatrix(data=X_validation, label = y_validation)
    
    final_validation_set = xgboost_validation
    
    roc(y_validation,
        predict(model, final_validation_set, type = "prob")[, "Positive"])
    
  } else{
    final_validation_set = validation_set
    
    roc(final_validation_set$RSV_test_result,
        predict(model, final_validation_set, type = "prob")[, "Positive"])  
  }
  
  
  
}

test_cm <- function(model, validation_data, threshold_pred = 0.5) {
  # Computes the confusion matrix for a binary classification model using a validation dataset.
  #
  # Inputs:
  #   model (fitted model object): The binary classification model to evaluate.
  #   validation_data (data frame): The validation dataset.
  #   threshold_pred (numeric): The probability threshold for predictions. Default is 0.5.
  #
  # Output:
  #   The confusion matrix for the model predictions.
  #
  # Example usage:
  #   test_cm(
  #     model = trained_model,
  #     validation_data = validation,
  #     threshold_pred = 0.6
  #   )
  
  actual_response = factor(ifelse(validation_data $RSV_test_result == "Positive",1,0))
  
  predictions_probs = predict(model, validation_data, type = "prob")[, "Positive"]
  predictions = factor(ifelse(predictions_probs > threshold_pred,1,0))
  
  caret::confusionMatrix(data = predictions, reference = actual_response, 
                         positive = "1")
  
}


# Functions to analyze sampling methods ####
train_smote_and_keep_levels = function(train_df, categorical_predictors,
                                       response_variable = "RSV_test_result",
                                       sampling_smote_list) {
  # Resamples the training data using SMOTE while keeping the levels of specified categorical predictors.
  #
  # Inputs:
  #   train_df (data frame): The training dataset to resample using SMOTE.
  #   categorical_predictors (vector of strings): The names of the categorical predictors whose levels need to be kept.
  #   response_variable (string): The target variable of the unbalanced data to deal with. Default is "RSV_test_result".
  #   sampling_smote_list (list): The list with attributes (name, func, first), same as the ones used for sampling in caret.
  #
  # Output:
  #   A data frame with the resampled training dataset with the original levels of the specified categorical predictors.
  #
  # Example usage:
  #   train_smote_and_keep_levels(
  #     train_df = train,
  #     categorical_predictors = c("col1", "col2", "col3"),
  #     response_variable = "RSV_test_result",
  #     sampling_smote_list = list(name = "smote", func = SMOTE, first = TRUE)
  #   )
  
  # One-hot encoding of predictors, as themis::smote does not admit categorical predictors
  train_df_onehot <- dummies::dummy.data.frame(train_df, names = categorical_predictors, sep = "_")
  X_train <- train_df_onehot[, !(colnames(train_df_onehot) %in% response_variable)]
  y_train <- train_df_onehot[, (colnames(train_df_onehot) %in% response_variable)]
  
  # Smote sampling according to the target sampling function
  smote_result = sampling_smote_list$func(x = X_train, y = y_train)
  X_train_smote <- smote_result$x
  y_train_smote <- smote_result$y
  
  X_train_smote_original <- X_train_smote
  
  # Replace the one-hot encoded columns with their original levels
  for (col in categorical_predictors) {
    levels <- unique(train_df[[col]])
    onehot_colnames <- colnames(X_train_smote_original)[startsWith(colnames(X_train_smote_original), paste0(col, "_"))]
    X_train_smote_original[[col]] <- levels[max.col(X_train_smote_original[onehot_colnames])]
    X_train_smote_original <- X_train_smote_original[, !colnames(X_train_smote_original) %in% onehot_colnames]
  }
  
  # Combine X_train_smote_original and y_train_smote to create the final balanced dataset
  train_df_smote <- cbind(X_train_smote_original, RSV_test_result = y_train_smote)
  
  return(train_df_smote)
  
}

combine_categorical_data <- function(original_data, smote_data, upsampled_data, downsampled_data, 
                                     categorical_columns) {
  # function to evaluate the impact of resampling following different methodologies
  # Two remarks
  # 1. The function is designed to handle 4 data frames: original, smote, upsampled and downsampled data
  # 2. This function only combines categorical data
  # 
  # Combines proportions of categorical data across different datasets and resampling methods.
  #
  # Inputs:
  #   original_data (data frame): The original dataset.
  #   smote_data (data frame): The dataset generated using SMOTE resampling method.
  #   upsampled_data (data frame): The dataset generated using upsampling resampling method.
  #   downsampled_data (data frame): The dataset generated using downsampling resampling method.
  #   categorical_columns (vector of strings): The names of the categorical columns in the datasets.
  #
  # Output:
  #   A data frame with the proportions of the categories in each dataset and resampling method.
  #
  # Example usage:
  #   combine_categorical_data(
  #     original_data = original,
  #     smote_data = smote,
  #     upsampled_data = upsampled,
  #     downsampled_data = downsampled,
  #     categorical_columns = c("col1", "col2", "col3")
  #   )
  
  combined_prop_all <- NULL
  
  for (col_name in categorical_columns) {
    original_prop <- original_data %>% 
      count(!!sym(col_name)) %>% 
      mutate(Proportion = n / sum(n), Dataset = "Original", Predictor = col_name)
    
    smote_prop <- smote_data %>% 
      count(!!sym(col_name)) %>% 
      mutate(Proportion = n / sum(n), Dataset = "Smote", Predictor = col_name)
    
    upsampled_prop <- upsampled_data %>% 
      count(!!sym(col_name)) %>% 
      mutate(Proportion = n / sum(n), Dataset = "Upsampled", Predictor = col_name)
    
    downsampled_prop <- downsampled_data %>% 
      count(!!sym(col_name)) %>% 
      mutate(Proportion = n / sum(n), Dataset = "Downsampled", Predictor = col_name)
    
    combined_prop <- bind_rows(original_prop, smote_prop, upsampled_prop, downsampled_prop) %>%
      select(Predictor, !!sym(col_name), n, Proportion, Dataset)
    
    colnames(combined_prop)[2] <- "Category"
    
    if (is.null(combined_prop_all)) {
      combined_prop_all <- combined_prop
    } else {
      combined_prop_all <- bind_rows(combined_prop_all, combined_prop)
    }
  }
  
  return(combined_prop_all)
}



#Save and load ML models in a database ####
save_model = function(folder_name = "~/R/models_store",
                      db_name = "trained_models.db", model, model_name) {
  # Saves a machine learning model to a database
  #
  # Inputs:
  #   db_name (string): The name of the database to save the trained model. Default value is "trained_models.db".
  #   model (trained model object): The machine learning model to save.
  #   model_name (string): The name to assign to the saved model.
  #
  # Example usage:
  #   save_model(
  #     db_name = "my_db",
  #     model = trained_model,
  #     model_name = "my_model"
  #   )
  current_folder = getwd()
  
  setwd(folder_name)
  
  #Connect to the database
  db_models = dbConnect(SQLite(), dbname = db_name, synchronous = NULL)
  
  # Adapt the model to a format able to load to a database
  model_serialized = serialize(model, NULL)
  df = data.frame(name = model_name, model = I(list(model_serialized)))
  
  #Load to the target database
  if (!dbExistsTable(db_models, "models")) {
    dbWriteTable(db_models, "models", df, row.names=FALSE)
  } else {
    # dbWriteTable(db_models, "models", df, row.names=FALSE, append=TRUE)
    dbAppendTable(db_models, "models", df)
  }
  
  print(paste0("Model ", model_name,": saved! at ",db_name))
  
  # Disconnect from the database
  dbDisconnect(db_models)
  
  setwd(current_folder)
}

load_model <- function(db_name = "trained_models.db", model_name) {
  # Loads a trained machine learning model from a database.
  #
  # Inputs:
  #   db_name (string): The name of the database containing the trained models. Default value is "trained_models.db".
  #   model_name (string): The name of the trained model to load from the database.
  #
  # Output:
  #   A trained machine learning model.
  #
  # Example usage:
  #   my_model <- load_model(model_name = "my_xgb_model")
  
  
  setwd("~/R/models_store")
  
  
  # Connect to database
  db_models <- dbConnect(SQLite(), dbname = db_name)
  
  # Query the model of interest
  res <- dbGetQuery(db_models, paste0("SELECT model FROM models WHERE name='", model_name, "'"))
  
  # Disconnect from database
  dbDisconnect(db_models)
  
  # Bring the model to the correct formal
  if (nrow(res) > 0) {
    res = unserialize(res$model[[1]])
  } else {
    stop("Model not found")
  }
  
  setwd("~/R/rwddia_434")
  
  return(res)
  
}


# Model performance and model fitting ####

train_model_and_no_more = function(ff, m,train_df,resampling_methods = c("original","weighted","down","up","SMOTE"),
                                   train_split = "8020"){
  
  # Trains a machine learning model using the specified algorithm and resampling methods on the training dataset `train_df`.
  # Uses cross-validation to tune hyperparameters using `trainControl`. Returns a trained machine learning model.
  #
  # Inputs:
  #   ff (formula): The formula object specifying the predictor variables and the response variable to be modeled.
  #   m (string): The name of the machine learning algorithm to use for modeling.
  #   train_df (data frame): The training dataset to be used for fitting the model.
  #   resampling_methods (vector of strings): The resampling methods to use for the model training. Can include "original", "weighted", "down", "up", "SMOTE".
  #   train_split (string): The proportion of the dataset to be used for training. Default value is "8020".
  #
  # Output:
  #   A trained machine learning model.
  #
  # Example usage:
  #   train_model_and_no_more(
  #     ff = RSV_test_result ~ .,
  #     m = "xgbTree",
  #     train_df = train_data,
  #     resampling_methods = c("original", "weighted", "down", "up", "SMOTE"),
  #     train_split = "8020"
  #   )
  
  
  resampling_methods = tolower(resampling_methods)
  
  tryCatch({
    if(length(resampling_methods) !=1){
      stop("The length of 'resampling_methods' is not equal to 1")
    } else {
      
      cl <- makePSOCKcluster(5)
      registerDoParallel(cl)
      
      # Here goes the actual method
      
      print(paste0("TRAINING with method: ",m))
      
      # Define the train_control within the for loop to ensure good seeds
      folds_cv = 10 # 10-folds cross-validation
      n_repeats_cv = 3
      
      train_control <- trainControl(method = "repeatedcv", number = 10,
                                    repeats = 3,
                                    summaryFunction = twoClassSummary,
                                    classProbs = T,
                                    allowParallel = T,
                                    search = "grid")
      
      
      if ("original" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: none"))
        
        # No rebalance
        orig_fit <- caret::train(ff, data = train_df, method = m,
                                 trControl = train_control, verbose = FALSE, metric = "ROC")
        model = orig_fit
      }
      
      if ("weighted" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: weigths"))
        
        # weighted classification
        model_weights <- ifelse(train_df$RSV_test_result == "Negative",
                                (1/table(train_df$RSV_test_result)[1]) * 0.5,
                                (1/table(train_df$RSV_test_result)[2]) * 0.5)
        #train_control$seeds <- orig_fit$control$seeds
        weighted_fit = caret::train(ff, data = train_df, method = m,
                                    trControl = train_control, verbose = FALSE, metric = "ROC",
                                    weights = model_weights)
        
        
        model = weighted_fit
        
      }
      
      if ("down" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: down"))
        
        # Down-sampled model
        train_control_xgb$sampling <- "down"
        
        down_fit = caret::train(ff, data = train_df, method = m,
                                trControl = train_control,verbose = FALSE, metric = "ROC")
        
        model = down_fit
      }
      
      if ("up" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: up"))
        print(train_control)
        # Up-sampled model
        train_control_xgb$sampling <- "up"
        
        up_fit = caret::train(ff, data = train_df, method = m,
                              trControl = train_control, verbose = FALSE, metric = "ROC")
        model = up_fit
      }
      
      if ("smote" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: SMOTE"))
        
        # SMOTE model
        train_control_xgb$sampling <- "smote"
        
        smote_fit = caret::train(ff, data = train_df, method = m,
                                 trControl = train_control,verbose = FALSE, metric = "ROC")
        model = smote_fit
      }
      
      stopCluster(cl)
      
      
      return(model)
    }
  }
  ,error = function(e) message("An error occurred: ", e$message)
  
  )
}



train_and_save_custom_xgb = function(ff, m = "xgbTree",train_df,resampling_methods = c("original","weighted","down","up","SMOTE"),
                                     db_store_name, train_split = "8020", tune_grid_xgb){
  resampling_methods = tolower(resampling_methods)
  
  # Trains and saves a machine learning model using the specified algorithm and resampling methods on the training dataset `train_df`.
  # Uses cross-validation to tune hyperparameters using `tune_grid_xgb`. Saves the trained models for each specified resampling method to a database.
  #
  # Inputs:
  #   ff (formula): The formula object specifying the predictor variables and the response variable to be modeled.
  #   m (string): The name of the machine learning algorithm to use for modeling. Default value is "xgbTree".
  #   train_df (data frame): The training dataset to be used for fitting the model.
  #   resampling_methods (vector of strings): The resampling methods to use for the model training. Can include "original", "weighted", "down", "up", "SMOTE".
  #   db_store_name (string): The name of the database to store the trained models.
  #   train_split (string): The proportion of the dataset to be used for training. Default value is "8020".
  #   tune_grid_xgb (data frame): The grid of hyperparameters to search through for tuning the model. If not provided, the function will use a default grid.
  #
  # Output:
  #   A trained machine learning model.
  #
  # Example usage:
  #   train_and_save_custom_xgb(
  #     ff = RSV_test_result ~ .,
  #     m = "xgbTree",
  #     train_df = train_data,
  #     resampling_methods = c("original", "weighted", "down", "up", "SMOTE"),
  #     db_store_name = "my_db",
  #     train_split = "8020",
  #     tune_grid_xgb = expand.grid(
  #       nrounds = c(100, 200, 300),
  #       max_depth = c(4, 6, 8, 10),
  #       eta = c(0.01, 0.05, 0.1),
  #       gamma = c(0, 0.5, 1),
  #       colsample_bytree = c(0.5, 0.7, 1),
  #       min_child_weight = c(1, 3, 5),
  #       subsample = c(0.5, 0.7, 1)
  #     )
  #   )
  
  tryCatch({
    if(length(resampling_methods) !=1){
      stop("The length of 'resampling_methods' is not equal to 1")
    } else {
      
      cl <- makePSOCKcluster(5)
      registerDoParallel(cl)
      
      # Here goes the actual method
      
      print(paste0("TRAINING with method: ",m))
      
      # Define the train_control within the for loop to ensure good seeds
      folds_cv = 10 # 10-folds cross-validation
      n_repeats_cv = 3
      
      # tune_grid_xgb <- expand.grid( nrounds = c(100, 200, 300), max_depth = c(4, 6, 8, 10),
      #                               eta = c(0.01, 0.05, 0.1), gamma = c(0, 0.5, 1),
      #                               colsample_bytree = c(0.5, 0.7, 1), min_child_weight = c(1, 3, 5),
      #                               subsample = c(0.5, 0.7, 1)      )
      
      train_control_xgb <- trainControl(
        method = "repeatedcv",
        number = 5, # 5-fold CV
        repeats = 3,
        summaryFunction = twoClassSummary,
        classProbs = TRUE,
        verboseIter = FALSE,
        returnData = FALSE,
        returnResamp = "all",
        savePredictions = TRUE
      )
      
      
      if ("original" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: none"))
        
        # No rebalance
        orig_fit <- caret::train(ff, data = train_df, method = m,
                                 trControl = train_control_xgb,
                                 tuneGrid = tune_grid_xgb, verbose = FALSE, metric = "ROC")
        model = orig_fit
        save_model(db_name = db_store_name, model = orig_fit, model_name = paste0("ff0_",m,"_original_",train_split ))
      }
      
      if ("weighted" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: weigths"))
        
        # weighted classification
        model_weights <- ifelse(train_df$RSV_test_result == "Negative",
                                (1/table(train_df$RSV_test_result)[1]) * 0.5,
                                (1/table(train_df$RSV_test_result)[2]) * 0.5)
        #train_control$seeds <- orig_fit$control$seeds
        weighted_fit = caret::train(ff, data = train_df, method = m,
                                    trControl = train_control_xgb,
                                    tuneGrid = tune_grid_xgb, verbose = FALSE, metric = "ROC",
                                    weights = model_weights)
        
        
        model = weighted_fit
        save_model(db_name = db_store_name, model = weighted_fit, model_name = paste0("ff0_",m,"_weighted_",train_split ))
        
      }
      
      if ("down" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: down"))
        
        # Down-sampled model
        train_control_xgb$sampling <- "down"
        
        down_fit = caret::train(ff, data = train_df, method = m,
                                trControl = train_control_xgb,
                                tuneGrid = tune_grid_xgb, verbose = FALSE, metric = "ROC")
        
        model = down_fit
        save_model(db_name = db_store_name, model = down_fit, model_name = paste0("ff0_",m,"_down_",train_split ))
      }
      
      if ("up" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: up"))
        print(train_control)
        # Up-sampled model
        train_control_xgb$sampling <- "up"
        
        up_fit = caret::train(ff, data = train_df, method = m,
                              trControl = train_control_xgb,
                              tuneGrid = tune_grid_xgb, verbose = FALSE, metric = "ROC")
        model = up_fit
        save_model(db_name = db_store_name, model = up_fit, model_name = paste0("ff0_",m,"_up_",train_split ))
      }
      
      if ("smote" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: SMOTE"))
        
        # SMOTE model
        train_control_xgb$sampling <- "smote"
        
        smote_fit = caret::train(ff, data = train_df, method = m,
                                 trControl = train_control_xgb,
                                 tuneGrid = tune_grid_xgb, verbose = FALSE, metric = "ROC")
        model = smote_fit
        save_model(db_name = db_store_name, model = smote_fit, model_name = paste0("ff0_",m,"_smote_",train_split ))
      }
      
      stopCluster(cl)
      
      
      return(model)
    }
  }
  ,error = function(e) message("An error occurred: ", e$message)
  
  )
}

train_custom_method_resampling = function(ff, m,train_df,resampling_methods = c("original","weighted","down","up","SMOTE")){
  resampling_methods = tolower(resampling_methods)
  
  # Trains a machine learning model using the specified algorithm and resampling 
  # methods on the training dataset `train_df`.
  # Evaluates the performance of the model on the validation dataset `validation_df` 
  # using the `actual_response` values.
  # Returns a list of two objects: `models` and `performances`. The `models` object 
  # is a list containing the trained models for each specified resampling method. 
  # The `performances` object is a list containing the performance metrics for each 
  # trained model.
  #
  # Inputs:
  #   ff (formula): The formula object specifying the predictor variables and the response variable to be modeled.
  #   m (string): The name of the machine learning algorithm to use for modeling.
  #   train_df (data frame): The training dataset to be used for fitting the model.
  #   validation_df (data frame): The validation dataset to be used for evaluating the model performance.
  #   actual_response (factor): The true response values for the validation dataset.
  #   resampling_methods (vector of strings): The resampling methods to use for the model training. Can include "original", "weighted", "down", "up", "SMOTE".
  #
  # Output: 
  #   A list containing two objects:
  #   - `models` (list): A list of trained models for each specified resampling method.
  #   - `performances` (list): A list of performance metrics for each trained model.
  
  tryCatch({
    if(length(resampling_methods) !=1){
      stop("The length of 'resampling_methods' is not equal to 1")
    } else {
      
      cl <- makePSOCKcluster(5)
      registerDoParallel(cl)
      
      # Here goes the actual method
      
      print(paste0("TRAINING with method: ",m))
      
      # Define the train_control within the for loop to ensure good seeds
      folds_cv = 10 # 10-folds cross-validation
      n_repeats_cv = 3
      
      if (m == "custom_rf"){
        train_control <- trainControl(method = "repeatedcv", number = 10,
                                      repeats = 3,
                                      summaryFunction = twoClassSummary,
                                      classProbs = T,
                                      allowParallel = T,
                                      search = "grid")
      }else{
        train_control <- trainControl(method = "repeatedcv", number = folds_cv,
                                      repeats = n_repeats_cv,
                                      summaryFunction = twoClassSummary,
                                      classProbs = T)
      }
      
      
      if ("original" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: none"))
        
        # No rebalance
        orig_fit <- caret::train(ff, data = train_df, method = m,
                                 verbose = FALSE, metric = "ROC", trControl = train_control)
        model = orig_fit
      }
      
      if ("weighted" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: weigths"))
        
        # weighted classification
        model_weights <- ifelse(train_df$RSV_test_result == "Negative",
                                (1/table(train_df$RSV_test_result)[1]) * 0.5,
                                (1/table(train_df$RSV_test_result)[2]) * 0.5)
        #train_control$seeds <- orig_fit$control$seeds
        weighted_fit <- caret::train(ff,  data = train_df, method = m, verbose = FALSE,
                                     weights = model_weights,
                                     metric = "ROC", trControl = train_control)
        
        model = weighted_fit
      }
      
      if ("down" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: down"))
        
        # Down-sampled model
        train_control$sampling <- "down"
        
        down_fit <- caret::train(ff, data = train_df,
                                 method = m, verbose = FALSE,
                                 metric = "ROC", trControl = train_control)
        model = down_fit
      }
      
      if ("up" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: up"))
        print(train_control)
        # Up-sampled model
        train_control$sampling <- "up"
        
        up_fit <- caret::train(ff, data = train_df,  method = m, verbose = FALSE,
                               metric = "ROC",  trControl = train_control)
        model = up_fit
      }
      
      if ("smote" %in% resampling_methods){
        print(paste0("Method: ",m," resampling: SMOTE"))
        
        # SMOTE model
        train_control$sampling <- "smote"
        
        smote_fit <- caret::train(ff, data = train_df,  method = m, verbose = FALSE,
                                  metric = "ROC",  trControl = train_control)
        model = smote_fit
      }
      
      stopCluster(cl)
      
      
      return(model)
    }
  }
  ,error = function(e) message("An error occurred: ", e$message)
  
  )
}

train_model_and_model_performance = function(ff, m,train_df, validation_df, actual_response,
                                             resampling_methods = c("original","weighted","down","up","SMOTE")){
  resampling_methods = tolower(resampling_methods)
  
  print(paste0("TRAINING with method: ",m))
  
  model_list <- vector(mode ="list", length = length(resampling_methods))
  names(model_list) = resampling_methods
  performance_list = model_list
  
  
  # Define the train_control within the for loop to ensure good seeds
  folds_cv = 10 # 10-folds cross-validation
  n_repeats_cv = 3
  train_control <- trainControl(method = "repeatedcv", number = folds_cv,
                                repeats = n_repeats_cv,
                                summaryFunction = twoClassSummary,
                                classProbs = T)
  
  if ("original" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: none"))
    
    # No rebalance
    orig_fit <- caret::train(ff, data = train_df, method = m,
                             verbose = FALSE, metric = "ROC", trControl = train_control)
    predictions_orig_fit = predict(orig_fit, newdata = validation_df, type = "raw")
    predictions_orig_fit = factor(ifelse(predictions_orig_fit == "Positive",1,0))
    performance_orig_fit = classification_metrics(predictions_orig_fit,actual_response)
    
    model_list $ original = orig_fit
    performance_list $ original = performance_orig_fit
  }
  
  if ("weighted" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: weigths"))
    
    # weighted classification
    model_weights <- ifelse(train_df$RSV_test_result == "Negative",
                            (1/table(train_df$RSV_test_result)[1]) * 0.5,
                            (1/table(train_df$RSV_test_result)[2]) * 0.5)
    #train_control$seeds <- orig_fit$control$seeds
    weighted_fit <- caret::train(ff,  data = train_df, method = m, verbose = FALSE,
                                 weights = model_weights,
                                 metric = "ROC", trControl = train_control)
    predictions_weighted_fit = predict(weighted_fit, newdata = validation_df, type = "raw")
    predictions_weighted_fit = factor(ifelse(predictions_weighted_fit == "Positive",1,0))
    performance_weighted_fit = classification_metrics(predictions_weighted_fit,actual_response)
    
    model_list $ weighted = weighted_fit
    performance_list $ weighted = performance_weighted_fit
  }
  
  if ("down" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: down"))
    
    # Down-sampled model
    train_control$sampling <- "down"
    #train_control$seeds <- orig_fit$control$seeds
    
    down_fit <- caret::train(ff, data = train_df,
                             method = m, verbose = FALSE,
                             metric = "ROC", trControl = train_control)
    predictions_down_fit = predict(down_fit, newdata = validation_df, type = "raw")
    predictions_down_fit = factor(ifelse(predictions_down_fit == "Positive",1,0))
    performance_down_fit = classification_metrics(predictions_down_fit,actual_response)
    
    model_list $ down = down_fit
    performance_list $ down = performance_down_fit
    
  }
  
  if ("up" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: up"))
    
    # Up-sampled model
    train_control$sampling <- "up"
    #train_control$seeds <- orig_fit$control$seeds
    
    up_fit <- caret::train(ff, data = train_df,  method = m, verbose = FALSE,
                           metric = "ROC",  trControl = train_control)
    predictions_up_fit = predict(up_fit, newdata = validation_df, type = "raw")
    predictions_up_fit = factor(ifelse(predictions_up_fit == "Positive",1,0))
    performance_up_fit = classification_metrics(predictions_up_fit,actual_response)
    
    model_list $ up = up_fit
    performance_list $ up = performance_up_fit
  }
  
  if ("smote" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: SMOTE"))
    
    # SMOTE model
    train_control$sampling <- "smote"
    #train_control$seeds <- orig_fit$control$seeds
    
    smote_fit <- caret::train(ff, data = train_df,  method = m, verbose = FALSE,
                              metric = "ROC",  trControl = train_control)
    predictions_smote_fit = predict(smote_fit, newdata = validation_df, type = "raw")
    predictions_smote_fit = factor(ifelse(predictions_smote_fit == "Positive",1,0))
    performance_smote_fit = classification_metrics(predictions_smote_fit,actual_response)
    
    model_list $ smote = smote_fit
    performance_list $ smote = performance_smote_fit
  }
  
  return(list(models = model_list,
              performances = performance_list))
}


# EDA FUNCTIONS #####
## functions to use along EDA
make_it_factor = function(df, is_phase1 = T){
  
  factor_columns = c("sex",                                                             
                     "race",                                                                  
                     "marital_status",                                                        
                     "patient_regional_location",                                             
                     "age_group",                                                             
                     "Conjuctivitis",                                                         
                     "Acute_upper_respiratory_infection",                                     
                     "Influenza"                                                             
                     ,"Pneumonia"                                                             
                     ,"Bronchitis"                                                            
                     ,"Acute_lower_respiratory_infection_other"                               
                     ,"Rhinitis"                                                              
                     ,"Other_COPD"                                                            
                     ,"Asthma"                                                                
                     ,"Symptoms_and_signs__circulatory_and_respiratory"                       
                     ,"Symptoms_and_signs__digestive_system_and_abdomen"                      
                     ,"Symptoms_and_signs__skin_and_subcutaneous_tissue"                      
                     ,"Symptoms_and_signs__cognition_perception_emotional_state_and_behaviour"
                     ,"General_symptoms_and_signs"                                            
                     ,"COVID19_related"                                                       
                     ,"any_symptom"                                                           
                     ,"Acute_myocardial_infarction"                                           
                     ,"Hystory_myocardial_infarction"                                         
                     ,"Congestive_heart_failure"                                              
                     ,"Peripheral_Vascular"                                                   
                     ,"CVD"                                                                   
                     ,"COPD"                                                                  
                     ,"Dementia"                                                              
                     ,"Paralysis"                                                             
                     ,"Diabetes"                                                              
                     ,"Diabetes_complications"                                                
                     ,"Renal_disease"                                                         
                     ,"mild_liver_disease"                                                    
                     ,"moderate_liver_disease"                                                
                     ,"Peptic_Ulcer_Disease"                                                  
                     ,"rheuma_disease"                                                        
                     ,"AIDS"                                                                  
                     ,"calendar_year"                                                         
                     ,"healthcare_seeking"                                                    
                     ,"influenza_vaccine",
                     "key_comorbidities",                                                     
                     "n_tests_that_day"
                     ,"any_immunodeficiency"
                     , 'tumor_indicator' 
                     ,'tumor_last_year' 
                     ,'is_metastatic')
  if (is_phase1){
    factor_columns = c('RSV_test_result', factor_columns)
  }
  
  #df[,names(df)[(sapply(df, class)) == "character"]] <- lapply(df[,names(df)[(sapply(df, class)) == "character"]] ,  factor)
  df[,factor_columns] = lapply(df[,factor_columns], factor)
  return(df)
}
center_and_scale = function(vector){
  # Takes a numeric vector as input and returns the vector centered and scaled.
  #
  # Inputs:
  #   vector (vector): The numeric vector to center and scale.
  #
  # Output: 
  #   A numeric vector containing the centered and scaled values of the input vector.
  
  out = (vector - mean(vector)) / sd(vector)
  return(out)
}

ggplot_positivity_demographics = function(demographics_df, category,
                                          title1 = paste0("Number of tests by ",category), 
                                          title2 = paste0("Distribution of positivity by ",category)){
  # RSV project: specific function
  # 
  # Creates two stacked bar plots using the `ggplot2` package to visualize the number 
  # of RSV tests and the distribution of positivity by demographic category.
  #
  # Inputs:
  #   demographics_df (data frame): The data frame containing the predictors and RSV test results.
  #   category (string): The name of the column in the data frame containing the demographic category to analyze.
  #   title1 (string): The title for the first plot (number of tests by category).
  #   title2 (string): The title for the second plot (distribution of positivity by category).
  #
  # Output: 
  #   A single grid containing two stacked bar plots:
  #     - The first plot shows the number of tests by demographic category.
  #     - The second plot shows the distribution of positivity by demographic category.
  
  
  
  # category is a string
  summarized_df <- demographics_df %>% 
    group_by(get(category)) %>% 
    summarize(total_count = n(RSV_test_result))
  names(summarized_df)[1] = category
  
  predictors_df_percent = demographics_df %>% 
    group_by(get(category), RSV_test_result) %>% 
    summarize(count_per_test = n(RSV_test_result) , .groups = 'keep') 
  names(predictors_df_percent)[1] = category
  predictors_df_percent = predictors_df_percent %>%
    left_join(y = summarized_df, by = category)%>%
    mutate(percent = (count_per_test / total_count))
  
  
  ymax = max(summarized_df $ total_count)*1.2
  
  p1 = ggplot(demographics_df) + aes(x = (get(category)), y = ..count..) + 
    geom_bar(aes(fill = RSV_test_result), position = position_stack(reverse = TRUE)) + 
    ylim(0,ymax) + xlab("") + 
    labs(title = title1)
  # p1 = p1 + geom_text(data = summarized_df, aes(x = get(category), y = total_count, label = total_count), 
  # position = position_dodge(0.9), vjust = -0.5, size = 3.5)
  
  p2 = ggplot(demographics_df) +
    aes(x = get(category), fill = RSV_test_result) +
    geom_bar(position = "fill") + xlab("")+
    labs(title = title2)
  p2 = p2 + geom_text(data = predictors_df_percent, aes(x = get(category), y = percent, 
                                                        label = paste0(round(percent*100,1), "%")), 
                      position = position_stack(vjust = 0.5), size = 2)
  p1 = p1 + coord_flip()
  p2 = p2 + coord_flip()
  
  grid.arrange(p1,p2)
}

id_overlap_table = function(l = list()){
  # Takes a list of id vectors and returns a table indicating the number and 
  # percentage of overlapping ids between each pair of vectors in the list.
  #
  # Inputs:
  #   l (list): A list of id vectors to check for overlap.
  #
  # Output: 
  #   A data frame where each cell contains the number of overlapping ids, 
  # with the percentage of overlap displayed as a comment.
  
  
  n_records = sapply(l, function(x) sapply(l, function(y) sum(y %in% x)))
  perc_records = sapply(l, function(x) sapply(l, function(y)( sum(y %in% x) / (ifelse(length(y)>=length(x), length(y),length(x)))  )))
  perc_records = round(perc_records*100)
  out = n_records
  out[lower.tri(out)] = perc_records[lower.tri(perc_records)]
  
  out = data.frame(out)
  
  
  pos_M = lower.tri(matrix(nrow = dim(out)[1],
                           ncol = dim(out)[2]))
  out [pos_M] = paste0(as.character(out[pos_M]), ' %')
  
  
  out
}

nas_table_int = function(df){
  # Calculates the percentage of missing and unknown values for each column in a specified data frame.
  #
  # Inputs:
  #   df (data frame): The data frame for which to calculate the percentage of missing and unknown values.
  #
  # Output: 
  #   A data frame with two columns:
  #     - `ratio_nas` (numeric): The percentage of missing values for each column in the data frame.
  #     - `ratio_unknowns` (numeric): The percentage of "unknown" values for each character column in the data frame.
  
  total_records = dim(df)[1]
  
  n_nas = ((sapply(df, function(x) sum(is.na(x)))) / total_records)*100
  n_unknowns = ((sapply(df, function(x) ifelse(class(x)=='character', sum(tolower(x) == 'unknown'), NA))) /total_records ) *100
  
  out = data.frame(ratio_nas = round(n_nas,1), ratio_unknowns = round(n_unknowns,1))
  out
}

find_top_codes = function(df, column_name = "code", n_top_codes = 10, decreasing = T){
  # Searches a specified data frame for the n_top_codes most frequently occurring 
  # codes in a specified column.
  #
  # Inputs:
  #   df (data frame): The data frame to search for top codes.
  #   column_name (character): The name of the column containing the codes to search for.
  #   n_top_codes (numeric): The number of top codes to return.
  #   decreasing (logical): Whether to return the top codes in decreasing order of frequency.
  #
  # Output: 
  #   A data frame with two columns:
  #     - `top_elements` (character): The top `n_top_codes` codes in the specified column.
  #     - `perc` (numeric): The percentage of times each code appears in the specified column.
  
  target_df = df %>% select(all_of(column_name))
  
  top_codes_df = data.frame(sort(table(target_df), decreasing = decreasing)[1:n_top_codes])
  names(top_codes_df)[1] = "top_elements"
  
  top_codes_df $ perc = round((top_codes_df $ Freq / dim(target_df)[1])*100, 1)
  
  # top_codes_df $ perc = sapply(top_encounter_codes $ perc, function(x) paste0(as.character(x), " %"))
  
  top_codes_df
}

kable_custom_table_from_df = function (df, caption_text = "") {
  # Converts a data frame to an HTML table with custom styling.
  #
  # Inputs:
  #   df (data frame): The data frame to convert to an HTML table.
  #   caption_text (character): The caption text to display above the table.
  #
  # Output: 
  #   An HTML table as a character string.
  
  kable(df,
        caption = caption_text,
        format.args = list(big.mark = ",")) %>%
    kable_styling(
      font_size = 15,
      bootstrap_options = c("striped", "hover", "condensed"),
      full_width = F,
      position = "center"
    ) 
  
}


# Legacy functions ####

calculate_fill_rate <- function(df){
  # Calculates the fill rate of a data frame, which is the percentage of non-missing values for each column in the data frame.
  #
  # Inputs:
  #   df (data frame): The data frame for which to calculate the fill rate.
  #
  # Output: 
  #   A list with two components:
  #     - `n_fill` (numeric): The number of non-missing values for each column in the data frame.
  #     - `perct_fill` (numeric): The fill rate (as a percentage) for each column in the data frame.
  
  calculate_
  
  n_fields = dim(df)[2]
  n_records = dim(df)[1]
  n_fill = numeric(n_fields)
  perct_fill = numeric(n_fields)
  
  for (i in 1:n_fields){
    n_missing = switch(class(df[[i]]),
                       "character" = max(sum(df[[i]] == ""), sum(is.na(df[[i]]))),
                       "integer" = sum(is.na(df[[i]])),
                       "logical" = sum(is.na(df[[i]])),
                       0)
    n_fill[i] = n_records - n_missing
    perct_fill [i] = n_fill[i] / n_records
  }
  return(list(n_fill = n_fill, perct_fill = perct_fill))
}



getCount <- function(db){
  # Retrieves the number of unique patients and total observations in a specified database table.
  #
  # Inputs:
  #   db (character): The name of the database table to count the number of patients and observations.
  #
  # Output: 
  #   A data frame with two columns:
  #     - `pts` (numeric): The number of unique patients in the specified database table.
  #     - `obs` (numeric): The total number of observations in the specified database table.
  
  dbGetQuery(con, glue("SELECT count(distinct patient_id) as pts, count(patient_id) as obs FROM {db} "))
  
}


drop_redshift <- function(tbl_name) {
  # Drops a Redshift table with the specified name.
  #
  # Inputs:
  #   tbl_name (character): The name of the Redshift table to drop.
  #
  # Output: None
  
  dbSendQuery(con, 
              glue("DROP TABLE IF EXISTS {tbl_name}"))
  
}

connect_to_database = function(database_name = "trinetx_rsv"){
  # Uses the `RocheData::get_data` function to retrieve a database connection 
  # object for the specified database name. It then calls the `connect()` method 
  # of the connection object to establish a connection to the database.
  #
  # Inputs:
  #   database_name (string): The name of the database to connect to.
  #
  # Output: 
  #   The function does not return any output, but it establishes a connection to the specified database.
  
  db <- RocheData::get_data(data = database_name)
  db$connect()
}

# 1. Setting up the project ####

load_codes_RSVburden = function(){
  # This function loads all the necessary codes in the RSV burden project.
  
  # Inputs: None
  # Outputs: A list of objects that contain the following codes:
  # - `codes_test_flu_like`: LOINC test codes encoding flu-like symptoms and RSV test
  # - `rsv_codes`: RSV test codes
  # - `codes_dx_flu_like`: Diagnosis codes encoding flu-like symptoms diagnoses
  # - `codes_dx_cs`: Diagnosis codes of target comorbidities
  # - `[condition]_dx_codes`: Diagnosis codes for each specific condition/comorbidity
  # - `codes_proced_cs`: Procedure codes of target comorbidities
  # - `[condition]_proced_codes`: Procedure codes for each specific condition/comorbidity
  # - `cpt_codes_influenza`: Influenza vaccine codes in CPT code system
  # - `rx_codes_influenza`: Influenza vaccine codes in RX code system
  
  setwd("~/R/rwddia_434")
  
  # 1. LOINC test codes encoding flu_like_symptoms and rsv test 
  flu_like_symptoms_test_codes <- read.csv("import/RSV_flu_like_symptoms_test_codes.csv")
  rsv_codes = flu_like_symptoms_test_codes $Code [flu_like_symptoms_test_codes $Condition == "RSV"]
  
  assign("codes_test_flu_like", flu_like_symptoms_test_codes, envir = .GlobalEnv)
  assign("rsv_codes", rsv_codes, envir = .GlobalEnv)
  
  
  # 2. Diagnosis codes encoding flu-like_symptoms diagnoses
  codes_dx_flu_like <- read.csv("import/RSV_flu_like_symptoms_dx_parent_codes.csv")
  codes_dx_flu_like$code_original <- codes_dx_flu_like$Parent.code
  codes_dx_flu_like $ code = codes_dx_flu_like $ code_original
  codes_dx_flu_like $ Parent.code = NULL
  codes_dx_flu_like $https...icd.who.int.browse10.2019.en. = NULL
  
  assign("codes_dx_flu_like", codes_dx_flu_like, envir = .GlobalEnv)
  
  
  # 3. codes_dx_cs: diagnosis codes of target comorbidities 
  codes_dx_cs <- read.csv("import/RSV_codes_dx_comorbidities.csv")
  codes_dx_cs $ code = codes_dx_cs $ Code
  codes_dx_cs $ Code = NULL
  assign("codes_dx_cs", codes_dx_cs, envir = .GlobalEnv)
  
  # # Loads the codes per condition / comorbidity
  # dx_conditions = unique(codes_dx_cs $Condition)
  # for (d in dx_conditions) {
  #   df <- sqldf(glue("SELECT code FROM codes_dx_cs WHERE Condition in ('{d}')"))
  #   
  #   assign(paste0(d,'_dx_codes'), df$code, envir = .GlobalEnv)
  # }
  
  
  # 4. codes_proced_cs: procedure codes of target comorbidities 
  codes_proced_cs <- read.csv("import/RSV_codes_procedure_comorbidities.csv")
  codes_proced_cs $ code = codes_proced_cs $ Code
  codes_proced_cs $ Code = NULL
  assign("codes_proced_cs", codes_proced_cs, envir = .GlobalEnv)
  
  # Loads every procedure code according to condition/comorbidity
  # proced_conditions = unique(codes_proced_cs $Condition)
  # for (p in proced_conditions) {
  #   df <- sqldf(glue("SELECT code FROM codes_proced_cs WHERE Condition in ('{p}')"))
  #   assign(paste0(p,'_proced_codes'),df$code, envir = .GlobalEnv)
  # }
  
  
  # 5. influenza vaccine codes
  # CPT: code system in the procedure table
  cpt_codes_influenza <- c('90630', '90654', '90653', '90655', '90656',
                           '90656', '90657', '90658', '90660',
                           '90662', '90672', '90673', '90674',
                           '90682', '90685', '90686', '90687',
                           '90688', '90689', '90694', '90756',
                           'Q2034','Q20345','Q2036','Q2037','Q2038','Q2039','G0008')
  assign('cpt_codes_influenza', cpt_codes_influenza, envir = .GlobalEnv)
  
  # RX: code system in the medication_drug table
  rx_codes_influenza <-c('1005911','1427022','1303855','1005931','1304122',
                         '1541617','857921','864701')
  assign('rx_codes_influenza', rx_codes_influenza, envir = .GlobalEnv)
  
  # 7. Immunodeficiencies codes
  
  immunodeficiencies_codes_df <- read.csv("import/RSV_immunodeficiency_icdcodes.csv")

  assign("immunodeficiencies_codes_df", immunodeficiencies_codes_df, envir = .GlobalEnv)

}


extract_positive_negative_all_patients_RSVburden = function(lab_result_RSV_df){
  # This function extracts the patient IDs of RSV positive patients, RSV negative patients, and all patients in a given RSV lab result data frame.
  
  # Inputs:
  # - `lab_result_RSV_df`: A data frame containing RSV lab results with columns `patient_id` and `lab_result_text_val`.
  
  # Outputs:
  # A list containing the following objects:
  # - `RSV_positive_patients`: A vector of patient IDs who tested positive for RSV.
  # - `RSV_negative_patients`: A vector of patient IDs who tested negative for RSV.
  # - `all_patients`: A vector of all patient IDs in the input data frame.
  
  RSV_positive_patients = unique(lab_result_RSV_df $ patient_id [lab_result_RSV_df $ lab_result_text_val == "Positive"])
  RSV_negative_patients = unique(lab_result_RSV_df $ patient_id [lab_result_RSV_df $ lab_result_text_val == "Negative"])
  all_patients = unique(lab_result_RSV_df $ patient_id)
  
  return(list(RSV_positive_patients = RSV_positive_patients,
              RSV_negative_patients = RSV_negative_patients,
              all_patients = all_patients))
}


load_phase1_predictors_df = function(){
  db <- RocheData::get_data(data = "trinetx_rsv")
  scratch_space = "scr_scr_449_rwddia434_rsv"
  
  db$schema = scratch_space
  db$connect()
  db$is_connected()
  conn = db$con
  
  rsv_predictors_df = dbReadTable(conn, "rsv_predictors_df_phase1") %>% make_it_factor()
  
  db$ disconnect()
  
  return(rsv_predictors_df)
}

load_phase2_predictors_df = function(){
  db <- RocheData::get_data(data = "trinetx_rsv")
  scratch_space = "scr_scr_449_rwddia434_rsv"
  
  db$schema = scratch_space
  db$connect()
  db$is_connected()
  conn = db$con
  
  rsv_predictors_df = dbReadTable(conn, "rsv_predictors_df_phase2") %>% make_it_factor(is_phase1 = FALSE)
  
  db$ disconnect()
  
  return(rsv_predictors_df)
}


# 1.2. Train_test_split ####
new_train_test_split_and_save = function(train_factor = 0.8, saving_train_test = F){
  "
  Function to produce a train/test split for ML training.
  Specific to the rsv_predictors_df dataframe, of the project on RSV_burden
  
  Input:
  train_factor (numeric) = number between 0.0 and 1.0 indicating the proportion of the data that goes to training
  saving_train_test (logical) = determines if the newly generated train_df and validation_df are saved to the scratch_space
  
  Output:
  list(train = train_df, validation_validation_df)
  "
  db <- RocheData::get_data(data = "trinetx_rsv")
  db$connect()
  db$is_connected()
  scratch_space = "scr_scr_449_rwddia434_rsv"
  db$schema = scratch_space
  conn = db$con
  
  #Load table from scratch space
  print("Loading rsv_predictors_df ...")
  rsv_predictors_df = dbReadTable(conn, "rsv_predictors_df")
  rsv_predictors_df[,names(rsv_predictors_df)[(sapply(rsv_predictors_df, class)) == "character"]] <- lapply(rsv_predictors_df[,names(rsv_predictors_df)[(sapply(rsv_predictors_df, class)) == "character"]] ,  factor)
  rsv_predictors_df = rsv_predictors_df %>%
    select(-c(patient_id, RSV_test_date))
  
  
  
  # 000. Train-test split ####
  # The very first step for everything is split data in train and validation sets
  # Very very first step
  
  print("Building train_df and validation_df ...")
  train_size = floor(train_factor*nrow(rsv_predictors_df))
  
  set.seed(102) # we set seed for reproducibility
  train_index <- sample(seq(1, nrow(rsv_predictors_df)), size = train_size)
  
  train_df = rsv_predictors_df[train_index,]
  validation_df = rsv_predictors_df[-train_index,]
  
  # train_df = train_df %>% select(-c(patient_id, RSV_test_date))
  # validation_df = validation_df %>% select(-c(patient_id, RSV_test_date))
  
  sum(train_df $ RSV_test_result == "Positive") / nrow(train_df) # 0.03059414
  sum(validation_df $ RSV_test_result == "Positive") / nrow(validation_df) # 0.03075237 
  
  
  # 0. Standarization of numerical variables ####
  # This part is to be done AFTER train-test split
  numeric_variables = names(rsv_predictors_df)[(sapply(rsv_predictors_df, class)) == "numeric" | 
                                                 (sapply(rsv_predictors_df, class)) == "integer" ]
  numeric_variables
  
  # #Normalize age
  # mean_age = mean(rsv_predictors_df $ age); sd_age = sd(rsv_predictors_df $ age)
  # rsv_predictors_df $ age = (rsv_predictors_df $ age - mean_age)/sd_age
  
  ## Normalize CCI
  additive_constant = 1
  # Train data 
  train_df = train_df %>%
    mutate(additive_logCCI = log(CCI + additive_constant))%>%
    select(-CCI)
  train_df $ additive_logCCI =  center_and_scale(train_df $ additive_logCCI)
  # validation data
  validation_df = validation_df %>%
    mutate(additive_logCCI = log(CCI + additive_constant))%>%
    select(-CCI)
  validation_df $ additive_logCCI = center_and_scale(validation_df $additive_logCCI)
  
  # # #Normalize calendar_week_number
  # # mean_week = mean(rsv_predictors_df $calendar_week_number); sd_week = sd(rsv_predictors_df $calendar_week_number)
  # # rsv_predictors_df[,'calendar_week_number'] = (rsv_predictors_df $calendar_week_number - mean_week)/ sd_week
  # 
  
  ## Standarize seasonality: sine and cosine
  train_df $ sine = center_and_scale(train_df $ sine)
  train_df $ cosine = center_and_scale(train_df $ cosine)
  
  validation_df $ sine = center_and_scale(validation_df $ sine)
  validation_df $ cosine = center_and_scale(validation_df $ cosine)
  
  
  # Save train_df and validation_df in scratch_space for reproducibility ####
  
  if (saving_train_test){
    print("Saving train_df ...")
    dbWriteTable(conn, paste0("train_df_",train_factor*100, 100*(1-train_factor)), train_df, 
                 overwrite = T)
    
    print("Saving validation_df ...")
    dbWriteTable(conn, paste0("validation_df_",train_factor*100, (1-train_factor)*100), validation_df, 
                 overwrite = T)
    
    print("Finished")
  }else{
    print("Train_df and validation_df not saved")
  }
  
  list(train = train_df, 
       validation = validation_df)
}


new_train_test_split = function(rsv_predictors_df, train_factor = 0.8){
  "
  Function to produce a train/test split for ML training.
  Specific to the rsv_predictors_df dataframe, of the project on RSV_burden
  
  Input:
  train_factor (numeric) = number between 0.0 and 1.0 indicating the proportion of the data that goes to training
  saving_train_test (logical) = determines if the newly generated train_df and validation_df are saved to the scratch_space
  
  Output:
  list(train = train_df, validation_validation_df)
  "
  
  # 000. Train-test split ####
  # The very first step for everything is split data in train and validation sets
  # Very very first step
  
  print("Building train_df and validation_df ...")
  train_size = floor(train_factor*nrow(rsv_predictors_df))
  
  set.seed(102) # we set seed for reproducibility
  train_index <- sample(seq(1, nrow(rsv_predictors_df)), size = train_size)
  
  train_df = rsv_predictors_df[train_index,]
  validation_df = rsv_predictors_df[-train_index,]
  
  # train_df = train_df %>% select(-c(patient_id, RSV_test_date))
  # validation_df = validation_df %>% select(-c(patient_id, RSV_test_date))
  
  sum(train_df $ RSV_test_result == "Positive") / nrow(train_df) # 0.03059414
  sum(validation_df $ RSV_test_result == "Positive") / nrow(validation_df) # 0.03075237 
  
  
  # 0. Standarization of numerical variables ####
  # This part is to be done AFTER train-test split
  numeric_variables = names(rsv_predictors_df)[(sapply(rsv_predictors_df, class)) == "numeric" | 
                                                 (sapply(rsv_predictors_df, class)) == "integer" ]
  numeric_variables
  
  # #Normalize age
  # mean_age = mean(rsv_predictors_df $ age); sd_age = sd(rsv_predictors_df $ age)
  # rsv_predictors_df $ age = (rsv_predictors_df $ age - mean_age)/sd_age
  
  ## Normalize CCI
  additive_constant = 1
  # Train data 
  train_df = train_df %>%
    mutate(additive_logCCI = log(CCI + additive_constant))%>%
    select(-CCI)
  train_df $ additive_logCCI =  center_and_scale(train_df $ additive_logCCI)
  # validation data
  validation_df = validation_df %>%
    mutate(additive_logCCI = log(CCI + additive_constant))%>%
    select(-CCI)
  validation_df $ additive_logCCI = center_and_scale(validation_df $additive_logCCI)
  
  # # #Normalize calendar_week_number
  # # mean_week = mean(rsv_predictors_df $calendar_week_number); sd_week = sd(rsv_predictors_df $calendar_week_number)
  # # rsv_predictors_df[,'calendar_week_number'] = (rsv_predictors_df $calendar_week_number - mean_week)/ sd_week
  # 
  
  ## Standarize seasonality: sine and cosine
  train_df $ sine = center_and_scale(train_df $ sine)
  train_df $ cosine = center_and_scale(train_df $ cosine)
  
  validation_df $ sine = center_and_scale(validation_df $ sine)
  validation_df $ cosine = center_and_scale(validation_df $ cosine)
  
  
  list(train = train_df, 
       validation = validation_df)
}



# Feature selection ####

print_auc = function(model_in, testData){
  predictions <- predict(model_in, newdata = testData, type = "prob")
  roc_model_ii <- roc(testData$RSV_test_result, predictions$Positive)
  print(auc(roc_model_ii))
}

unlikeability_fun = function(factor_vector){
  
  if (class(factor_vector) == "factor"){
    
    p_i = table(factor_vector) / length(factor_vector)
    
    unlikeability = sum(p_i*(1-p_i))
    
    return(unlikeability)
    
  }else{
    stop("input is not a factor! cannot compute its unlikeability")
  }
}

# Feature build-up ####

create_n_symptoms = function(df, flu_like_symptoms){
  aux_df = df %>%
    select(all_of(flu_like_symptoms)) %>%
    mutate(n_symptoms = apply(., 1, function(x) sum(as.numeric(x))))
  df $ n_symptoms = aux_df $ n_symptoms
  
  return(df)
}

create_n_encounters = function(df){ 
  df = df %>%
    left_join(y = df %>% count(patient_id), by = 'patient_id') %>%
    rename(n_encounters = n)
  return(df)
}


create_key_comorbidities = function(df,
                                    key_comorbs = c('COPD', 'Asthma')){
  
  
  
  df $ key_comorbidities = as.factor((df %>%
                                        select(all_of(key_comorbs)) %>%
                                        mutate(key_comorbidities = apply(., 1, function(x) sum(as.numeric(x)))) %>%
                                        select(key_comorbidities)) $ key_comorbidities)
  return(df)
}

detect_prev_positive_rsv_tests = function(rsv_test_df){
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (RSV_test_date = index_date)
  }  
  
  rsv_test_df = rsv_test_df %>%
    arrange(patient_id, RSV_test_date) %>%
    group_by(patient_id) %>%
    mutate(prev_positive_rsv = cumsum(ifelse(lag(RSV_test_result, default = "Negative") == "Positive", 1, 0)))
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (index_date = RSV_test_date)
  } 
  return(rsv_test_df)
}


detect_tendency_to_positivity = function(rsv_test_df){
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (RSV_test_date = index_date)
  } 
  
  out_df = rsv_test_df %>%
    group_by(patient_id) %>%
    mutate(tendency_to_positivity = sum(ifelse(RSV_test_result == "Positive", 1, 0)) / sum(ifelse(RSV_test_result == 'Positive', 1,1)) )%>%
    ungroup()
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (index_date = RSV_test_date)
  } 
  
  return(out_df)
}


detect_previous_test_day_diff = function(rsv_test_df){
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (RSV_test_date = index_date)
  }
  
  aux_df = rsv_test_df %>%
    arrange(patient_id, RSV_test_date, RSV_test_result) %>%
    group_by(patient_id)%>%
    mutate(
      prev_date = lag(RSV_test_date, default = NA),
      previous_test_daydiff = as.numeric(difftime(RSV_test_date, prev_date, days)),
    ) %>%
    select(-c(prev_date)) %>%
    replace(is.na(.),365*3) %>%
    ungroup()
  
  rsv_test_df $ previous_test_daydiff = aux_df $ previous_test_daydiff
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (index_date = RSV_test_date)
  } 
  
  return(rsv_test_df)
}

detect_n_tests_that_day = function(rsv_test_df){
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (RSV_test_date = index_date)
  }
  
  aux_df = rsv_test_df %>%
    select(patient_id, RSV_test_date) %>%
    group_by(patient_id, RSV_test_date) %>%
    count()
  aux2_df = rsv_test_df %>%
    left_join(y = aux_df, by = c('patient_id', 'RSV_test_date'))
  
  rsv_test_df $n_tests_that_day = factor(aux2_df $ n, levels = c(0:max(aux2_df $ n)))
  
  if ("RSV_test_date" %in% names(rsv_test_df) == FALSE){
    rsv_test_df = rsv_test_df %>% rename (index_date = RSV_test_date)
  } 
  
  return(rsv_test_df)
}



demographics_df_build_up_RSVburden = function(predictors_df, patient_demographic_df){
  
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  # This function incorporates demographic information from the patient_demographic_df 
  # data frame into the predictors_df data frame. It performs various transformations 
  # and calculations to build up a new data frame called predictors_df_dem.
  # 
  # Inputs:
  # - predictors_df: The input data frame containing predictors or features. It needs to have the following columns:
  #   'patient_id', 'index_date', 'RSV_test_result' (optional)
  # - patient_demographic_df: The data frame containing patient demographic information, 
  #   including columns like "patient_id," "sex," "race," "ethnicity," "marital_status," "patient_regional_location," "RSV_test_date," and "year_of_birth."
  # Output:
  # - predictors_df_dem, which includes the original predictors along with additional columns:
  #   "sex," "race", "age_group", "marital_status", "patient_regional_location"
  
  
  assign("predictors_df_dem", predictors_df)
  
  predictors_df_dem = predictors_df %>%
    left_join (y = patient_demographic_df, by = "patient_id") %>%
    mutate(sex = as.factor(sex))%>% 
    mutate(race = as.factor(race))%>%
    mutate(ethnicity = as.factor(ethnicity))%>% 
    mutate(marital_status = as.factor(marital_status))%>%
    mutate(patient_regional_location = as.factor(patient_regional_location))%>%
    mutate(age = as.numeric(format(index_date, "%Y")) - year_of_birth) %>%
    mutate (age_group = factor(case_when(age >= 18  & age <= 25 ~ '18-25',
                                         age >= 26  & age <= 30 ~ '26-30',
                                         age >= 31  & age <= 35 ~ '31-35',
                                         age >= 36  & age <= 40 ~ '36-40',
                                         age >= 41  & age <= 45 ~ '41-45',
                                         age >= 46  & age <= 50 ~ '46-50', 
                                         age >= 51  & age <= 55 ~ '51-55', 
                                         age >= 56  & age <= 60 ~ '56-60',
                                         age > 60 ~ '>60',
                                         TRUE ~ 'NA'), 
                               levels = c('18-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','>60','NA'))
            ,
            race = factor(case_when(race == 'White' ~ 'White',
                                    race == 'Black or African American' ~ 'Black',
                                    race == 'Asian' ~ 'Asian',
                                    race == 'Native Hawaiian or Other Pacific Islander' ~ 'Pacific', 
                                    race == 'American Indian or Alaska Native' ~ 'Native American',
                                    race == 'Unknown' ~ 'Unknown'), 
                          levels = c('White','Black','Asian', 'Pacific','Native American','Unknown'))) %>%
    filter((age >= 18) & (age <= 60)) %>%
    # filter(age >= 18) %>%
    select(-c(reason_yob_missing, month_year_death, death_date_source_id, source_id,
              ethnicity, year_of_birth, age))
  # select(-c(reason_yob_missing, month_year_death, death_date_source_id, source_id,
  #           ethnicity, year_of_birth))
  
  return(predictors_df_dem)
  
}


tumor_df_build_up_RSVburden = function(predictors_df, dedup_tumor_df) {
  
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  # Description:
  #
  # Inputs:
  # - predictors_df: 
  # - dedup_tumor_df: T
  # Outputs:
  # - predictors_df_tumor:
  
  
  assign("predictors_df_tumor", predictors_df)
  
  aux_tumor_df = dedup_tumor_df %>%
    select(patient_id, observation_date, metastatic) %>%
    mutate(metastatic = as.factor(replace_na(metastatic, "0")))
  
  
  aux_df =  predictors_df_tumor%>%
    left_join(y = aux_tumor_df, by = 'patient_id', multiple = 'all') %>%
    mutate(time_difference = difftime(observation_date, index_date, units = "days" )) %>% # date_tumor - RSV_test date 
    filter( time_difference <= 0) %>% # we take into account any date prior to the rsv test date
    mutate(tumor_indicator = 1) %>%
    mutate( tumor_last_year = (ifelse(  (time_difference >= -365)&(time_difference <= 0),1,0)) ) %>%
    mutate(is_metastatic = (ifelse( metastatic == 0,0,1  ))) %>%
    select(patient_id, index_date, tumor_indicator, tumor_last_year, is_metastatic) %>%
    group_by(patient_id, index_date) %>%
    mutate(tumor_last_year = ( ifelse(any(tumor_last_year == 1) ,1,0))) %>%
    mutate(is_metastatic = ( ifelse(any(is_metastatic == 1) ,1,0))) %>%
    ungroup() %>%
    distinct()
  
  # Finally, merge the immuno_df with previous 
  
  predictors_df_tumor <- predictors_df %>% 
    left_join(y = aux_df, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    replace(is.na(.),0) %>%
    mutate(tumor_indicator = as.factor(tumor_indicator),
           tumor_last_year = as.factor(tumor_last_year),
           is_metastatic = as.factor(is_metastatic),)
  
  return(predictors_df_tumor)
}


immunodeficiency_df_build_up_RSVburden = function(predictors_df, dedup_diagnosis_df,
                                                   immunodeficiency_df = immunodeficiencies_codes_df) {
  
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  # Description:
  #
  # Inputs:
  # - predictors_df: 
  # - dedup_diagnosis_df: T
  # - immunodeficiency_df
  # Outputs:
  # - predictors_df_immuno:
  
  
  assign("predictors_df_immuno", predictors_df)
  
  # Flu-like symptoms are encoded using diagnosis codes
  # Hence loading all diagnosis codes is needed
  
  
  aux_diagnosis = dedup_diagnosis_df %>%
    mutate(sub_code_3 = substring(code, 1, 3)) %>%
    mutate(sub_code_5 = substring(code, 1, 5)) %>%
    mutate(sub_code_6 = substring(code, 1, 6)) %>%
    mutate(sub_code_7 = substring(code, 1, 7)) %>%
    mutate(date = as.Date(date, format = "%Y-%m-%d"))
  
  # 2. Filter in all patients presenting a given immunodeficiency
  #Immunodeficiencies of interest are taken in any time prior to the RSV test date  
  immunodeficiencies = unique(immunodeficiency_df $Condition)
  codes_immuno = immunodeficiency_df $ Code
  
  aux_diagnosis <- aux_diagnosis %>%
    filter(sub_code_3 %in% codes_immuno | 
             sub_code_5 %in% codes_immuno | 
             sub_code_6 %in% codes_immuno | 
             sub_code_7 %in% codes_immuno)
  
  patient_immuno_list <- list()
  
  
  for (ii in 1:length(immunodeficiencies)){
    Im = immunodeficiencies[ii]
    codes_Im = immunodeficiency_df$Code[immunodeficiency_df $ Condition == Im]
    
    aux_df <- predictors_df_immuno %>% 
      inner_join(y = aux_diagnosis%>%
                   filter(sub_code_3 %in% codes_Im | 
                            sub_code_5 %in% codes_Im | 
                            sub_code_6 %in% codes_Im | 
                            sub_code_7 %in% codes_Im), by ='patient_id', keep = FALSE, multiple = 'all') %>%
      mutate(time_difference = difftime(date, index_date, units = "days" )) %>% # date_immnodeficiency_diagnosis - RSV_test date 
      filter( time_difference <= 0) %>% # we take into account any date prior to the rsv test date
      select(patient_id, index_date) %>%
      distinct()
    
    if(sum(!is.empty( aux_df $ patient_id)) > 0) {
      patient_immuno_df <- data.frame(patient_id = aux_df$patient_id, index_date = aux_df $index_date, 
                                      immunodeficiency = Im)
      
      patient_immuno_list[[Im]] <- patient_immuno_df
    }
    
  }
  
  # 3. Compute the number of immunodeficiencies every patient has 
  
  all_patient_immuno <- do.call(rbind, patient_immuno_list)
  
  patient_immuno_counts <- all_patient_immuno %>%
    group_by(patient_id, index_date) %>%
    summarise(n_immunodeficiencies = n_distinct(immunodeficiency), .groups = 'drop')
  
  # Finally, merge the immuno_df with previous 
  
  predictors_df_immuno <- predictors_df %>% 
    left_join(y = patient_immuno_counts, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    replace(is.na(.),0) %>%
    mutate(any_immunodeficiency = as.factor(ifelse(n_immunodeficiencies > 0 , 1,0 )))
  
  return(predictors_df_immuno)
}




flu_like_symptoms_df_build_up_RSVburden = function(predictors_df, dedup_diagnosis_df,
                                                   window_flu_symptoms = 7, 
                                                   flu_codes_dx = codes_dx_flu_like) {
  
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  # Description: This function incorporates flu-like symptom information into the predictors_df data frame based on diagnosis codes. It creates two sets of data frames: one with time differences between the symptom date and index date, and another with binary indicators for the presence of flu-like symptoms. Finally, it adds the flu-like symptom information to the predictors_df data frame and returns the modified predictors_df_flu_like data frame.
  # Inputs:
  # - predictors_df: The input data frame containing predictors or features. It needs to have the following columns:
  #   'patient_id', 'index_date', 'RSV_test_result' (optional)
  # - dedup_diagnosis_df: The data frame containing deduplicated diagnosis information, 
  #   including columns like "patient_id," "code," and "date."
  # - window_flu_symptoms: The time window (in days) around the index date to consider for flu-like symptoms (default: 7).
  # - flu_codes_dx: A dataframe of flu-like diagnosis codes.  Generated calling load_codes_RSVburden()
  # Outputs:
  # - predictors_df_flu_like, which includes the original predictors along with 15
  #    additional columns indicating the presence of 14 specific flu-like symptoms 
  #    and an "any_symptom" column indicating the presence of any flu-like symptom.
  
  
  assign("predictors_df_flu_like", predictors_df)
  # 7 days pre and post index data for flu-like symptoms relevance
  
  
  # Flu-like symptoms are encoded using diagnosis codes
  # Hence loading all diagnosis codes is needed
  
  assign("aux_diagnosis", dedup_diagnosis_df)
  aux_diagnosis = aux_diagnosis %>%
    mutate(sub_code_3 = substring(code, 1, 3)) %>%
    mutate(sub_code_5 = substring(code, 1, 5)) %>%
    mutate(date = as.Date(date, format = "%Y-%m-%d"))
  
  
  # 2. Filter in all patients presenting a given flu-like symptom
  # 2a) This first df incorporates the time difference (index_date - date of the symptom)
  
  flu_like_symptoms = unique(flu_codes_dx $Predictor)
  
  for (ii in 1:length(flu_like_symptoms)){
    f = flu_like_symptoms[ii]
    codes_f = flu_codes_dx$code[flu_codes_dx $ Predictor == f]
    
    aux_df <- predictors_df_flu_like %>% 
      inner_join(y = aux_diagnosis[ (aux_diagnosis$sub_code_3 %in% codes_f)|(aux_diagnosis$sub_code_5 %in% codes_f), ], 
                 by ='patient_id', keep = FALSE, multiple = 'all') %>%
      mutate(time_difference = difftime(date, index_date, units = "days" )) %>% # date_symptom - RSV_test date 
      filter( abs(time_difference) <= window_flu_symptoms) 
    
    assign(paste0(f,"_patients_df_dt"), aux_df)
  }
  
  # 2b. This second df just adds a binary variable 1/0 to indicate if the flu_like symptom is present or not
  for (ii in 1:length(flu_like_symptoms)){
    f = flu_like_symptoms[ii]
    df = get(paste0(f,"_patients_df_dt"))
    
    aux_df <- df %>% 
      group_by(patient_id, index_date)  %>%
      summarize(indicator = 1, .groups = 'keep') 
    
    names(aux_df)[3] = f
    
    assign(paste0(f,"_patients_df"), aux_df)
  }
  
  
  predictors_df_flu_like <- predictors_df %>% 
    left_join(y = Conjuctivitis_patients_df, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `Acute upper respiratory infection_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Influenza_patients_df, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Pneumonia_patients_df, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Bronchitis_patients_df, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `Acute lower respiratory infection (other)_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Rhinitis_patients_df, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `Other COPD_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Asthma_patients_df, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `Symptoms and signs - circulatory and respiratory_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `Symptoms and signs - digestive system and abdomen_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `Symptoms and signs - skin and subcutaneous tissue_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `Symptoms and signs - cognition, perception, emotional state and behaviour_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `General symptoms and signs_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = `COVID-19 related_patients_df`, by =c('patient_id', 'index_date'), keep = FALSE) %>% 
    replace(is.na(.),0) 
  predictors_df_flu_like$ any_symptom <- apply(predictors_df_flu_like[,flu_like_symptoms], 1, function(x) ifelse(sum(x==1) == 0, 0,1) )
  predictors_df_flu_like[,c(flu_like_symptoms,"any_symptom")] <- lapply(predictors_df_flu_like[,c(flu_like_symptoms,"any_symptom")] , factor)
  
  
  return(predictors_df_flu_like)
}


comorbidities_df_build_up_RSVburden = function(predictors_df, dedup_diagnosis_df, dedup_procedure_df,
                                               threshold_cs = 0,
                                               dx_comorbidities_df = codes_dx_cs,
                                               proced_comorbidities_df = codes_proced_cs){
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  # Description: This function incorporates comorbidity information into the predictors_df data frame based on diagnosis and procedure codes. It creates individual data frames for each comorbidity category, including those determined by symptoms (diagnosis codes) and procedures. It then combines the individual data frames and adds the comorbidity information to the predictors_df data frame, returning the modified predictors_df_cs data frame.
  # Inputs:
  # - predictors_df: The input data frame containing predictors or features. It needs to have the following columns:
  #   'patient_id', 'index_date', 'RSV_test_result' (optional)
  # - dedup_diagnosis_df: The data frame containing deduplicated diagnosis information, including columns like "patient_id," "code," and "date."
  # - dedup_procedure_df: The data frame containing deduplicated procedure information, including columns like "patient_id," "code," and "date."
  # - threshold_cs: The time threshold (in days) before the index test to consider for comorbidities (default: 0).
  # - dx_comorbidities_df: A data frame specifying comorbidities determined by symptoms (diagnosis codes).
  # - proced_comorbidities_df: A data frame specifying comorbidities determined by procedures.
  # Outputs:
  # - predictors_df_cs, which includes the original predictors along with 17 additional columns indicating the 
  # presence of specific comorbidities and a "CCI" (Charlson Comorbidity Index) column indicating the total number of comorbidities.
  
  
  # only take into account comorbidities before the RSV test takes place
  
  
  # initialize the df
  assign("predictors_df_cs", predictors_df)
  
  assign("aux_diagnosis", dedup_diagnosis_df)
  aux_diagnosis = aux_diagnosis %>%
    mutate(sub_code_3 = substring(code, 1, 3)) %>%
    mutate(sub_code_5 = substring(code, 1, 5)) %>%
    mutate(sub_code_6 = substring(code, 1, 6)) %>%
    mutate(date = as.Date(date, format = "%Y-%m-%d"))
  
  assign("aux_procedure", dedup_procedure_df)
  aux_procedure = aux_procedure %>%
    mutate(sub_code_3 = substring(code, 1,3))%>%
    mutate(sub_code_4 = substring(code, 1,4))%>%
    mutate(sub_code_5 = substring(code, 1,5))%>%
    mutate(date = as.Date(date, format = "%Y-%m-%d"))
  
  
  # Build-up of individual df's capturing comorbidites 1 by 1
  
  # 1st. comorbidities determined by symptoms
  
  dx_conditions = unique(dx_comorbidities_df $ Condition)
  
  for (d in dx_conditions){
    
    codes_d = (dx_comorbidities_df %>% filter (Condition == d) %>% select(code)) $ code
    
    aux_df <- predictors_df %>% 
      inner_join(y = aux_diagnosis[ (aux_diagnosis$sub_code_3 %in% codes_d)|
                                      (aux_diagnosis$sub_code_5 %in% codes_d)|
                                      (aux_diagnosis$sub_code_6 %in% codes_d), ], 
                 by ='patient_id', keep = FALSE, multiple = 'all') %>%
      mutate(time_difference = difftime(date, index_date, units = "days" )) %>% # date_comorbidity - index_date 
      filter( time_difference <= threshold_cs) %>% 
      group_by(patient_id, index_date)  %>%
      summarize(indicator = 1, .groups = 'keep') 
    
    names(aux_df)[3] = d
    
    assign(paste0(d,"_patients_cs_df"), aux_df)
  }
  
  # 2nd. comorbidities determined by procedures
  proced_conditions = unique(proced_comorbidities_df $ Condition)
  
  for (p in proced_conditions){
    codes_p = (proced_comorbidities_df %>% 
                 filter(Condition == p) %>% select(code)) $ code
    
    aux_df <- predictors_df %>% 
      inner_join(y = aux_procedure[ (aux_procedure$sub_code_3 %in% codes_p)|
                                      (aux_procedure$sub_code_5 %in% codes_p) |
                                      (aux_procedure $ sub_code_4 %in% codes_p), ], 
                 by ='patient_id', keep = FALSE, multiple = 'all') %>%
      mutate(time_difference = difftime(date, index_date, units = "days" )) %>% # date_comorbidity - index_date 
      filter(time_difference <= threshold_cs) %>% 
      group_by(patient_id, index_date)  %>%
      summarize(indicator = 1, .groups = 'keep') 
    
    names(aux_df)[3] = p
    
    assign(paste0(p, "_patients_proced_cs_df"), aux_df)
    
  }
  
  # 3rd: some comorbidities are present both in procedures and in diagnosis, we need to keep both
  overlapping_conditions = intersect(dx_conditions, proced_conditions)
  
  for (o in overlapping_conditions){
    
    df_dx = get(paste0(o,"_patients_cs_df"))
    df_proc = get(paste0(o,"_patients_proced_cs_df"))
    
    aux_df = df_dx %>%
      full_join(y = df_proc, by = c('patient_id','index_date') ) %>%
      replace(is.na(.),0) 
    names(aux_df)[3] = "indicator.dx"
    names(aux_df)[4] = "indicator.proc"
    
    aux_df = aux_df %>%
      mutate(indicator_total = sum(indicator.dx + indicator.proc)) %>%
      filter(indicator_total > 0) %>%
      summarize(indicator = 1, .groups = 'keep') 
    names(aux_df)[3] = o
    
    assign(paste0(o,"_patients_cs_df"), aux_df)
  }
  
  
  predictors_df_cs <- predictors_df %>% 
    left_join(y = Acute_myocardial_infarction_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Hystory_myocardial_infarction_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Congestive_heart_failure_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Peripheral_Vascular_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = CVD_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = COPD_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Dementia_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Paralysis_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Diabetes_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Diabetes_complications_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Renal_disease_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = mild_liver_disease_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = moderate_liver_disease_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Peptic_Ulcer_Disease_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = rheuma_disease_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = AIDS_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>% 
    left_join(y = Asthma_chronic_patients_cs_df, by = c('patient_id', 'index_date'), keep = FALSE) %>%
    replace(is.na(.),0) 
  predictors_df_cs $ CCI <- apply(predictors_df_cs[,dx_conditions], 1, function(x) sum(x==1) )
  predictors_df_cs[,dx_conditions] <- lapply(predictors_df_cs[,dx_conditions] , factor)
  
  return(predictors_df_cs)
}

seasonality_df_buildup_RSVburden = function(predictors_df){
  
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  #
  # Description: This function adds a seasonality predictor to the predictors_df data frame. 
  # It includes the calendar year of the test and a periodic element characterized by sine 
  # and cosine functions. The resulting predictors_df_season data frame contains the original 
  # predictors along with additional columns for calendar year, sine, and cosine values.
  # 
  # Inputs:
  # - predictors_df: The input data frame containing predictors or features. It needs to have the following columns:
  #   'patient_id', 'index_date', 'RSV_test_result' (optional)
  # 
  # Outputs:
  # - predictors_df_season: includes the original predictors along with additional columns for calendar year, sine values, and cosine values.
  
  
  # 3.1 Add calendar year (exclusively)
  wavelength = 52 # a calendar year has 52 weeks
  assign("predictors_df_season", predictors_df)
  
  predictors_df_season = predictors_df %>%
    mutate(calendar_week_number = lubridate::isoweek(index_date)) %>%
    mutate(sine = sin((2*pi*calendar_week_number)/wavelength )) %>%
    mutate(cosine = cos((2*pi*calendar_week_number)/wavelength)) %>%
    mutate(calendar_year = format(index_date, format = "%Y")) %>%
    select(-c('calendar_week_number'))
  predictors_df_season[,'calendar_year'] <- as.factor(predictors_df_season[,'calendar_year'])
  
  return(predictors_df_season)
}

healthcare_seeking_df_buildup_RSVburden = function(predictors_df,
                                                   threshold_encounters = 2,
                                                   RSV_season_day_one = "11-01",
                                                   RSV_season_last_day = "04-30"){
  
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  #
  # # Description: This function identifies healthcare seekers based on the number of tests conducted within a specific season. 
  # It takes into account the threshold for the number of tests, the RSV season start and end dates, and the predictors_df data 
  # frame. It creates a healthcare_seeking_df data frame that includes the patient ID, index date, and a healthcare_seeking 
  # column indicating whether the patient is a healthcare seeker or not.
  # 
  # Inputs:
  # - predictors_df: The input data frame containing predictors or features. It needs to have the following columns:
  #   'patient_id', 'index_date', 'RSV_test_result' (optional)
  # - threshold_encounters: The threshold for the number of tests within a season to consider a patient as a healthcare seeker (default: 2).
  # - RSV_season_day_one: The start date (month-day format) of the RSV season (default: "11-01", i.e. November 1st).
  # - RSV_season_last_day: The end date (month-day format) of the RSV season (default: "04-30", i.e. April 30th).
  # 
  # Outputs:
  # - healthcare_seeking_df (data frame): includes the patient ID, index date, and a healthcare_seeking column indicating
  #   whether the patient is a healthcare seeker (1) or not (0).
  
  
  # every patient who's had >=thresold tests is considered to be healthcare seeker
  # this threshold is established to be in the same season
  winter_start = lubridate::isoweek(as.Date(RSV_season_day_one, format = "%m-%d"))
  winter_end = lubridate::isoweek(as.Date(RSV_season_last_day, format = "%m-%d"))
  
  healthcare_seeking_df = predictors_df %>%
    mutate(calendar_week_number = lubridate::isoweek(index_date)) %>%
    mutate(calendar_year = format(index_date, format = "%Y")) %>%
    mutate(season_of_test = factor(case_when((calendar_week_number < winter_end)  | (calendar_week_number >= winter_start) ~ 'Flu season', 
                                             calendar_week_number >= winter_end & calendar_week_number < winter_start ~ 'Off season'),
                                   levels = c('Flu season','Off season')) ) %>%
    mutate(actual_season = factor(case_when( season_of_test == 'Flu season' & calendar_week_number < winter_end ~ paste0('Season_',as.numeric(calendar_year) - 1, calendar_year),
                                             season_of_test == 'Flu season' & calendar_week_number >= winter_start ~ paste0('Season_',calendar_year,as.numeric(calendar_year)+1),
                                             season_of_test == 'Off season' ~ paste0('Off_season_',calendar_year))))
  
  df_counts = as.data.frame(table(healthcare_seeking_df $patient_id, healthcare_seeking_df $ actual_season))
  names(df_counts) = c("patient_id","actual_season","count")
  
  df_counts = df_counts %>% 
    group_by(patient_id)%>%
    mutate(max_tests_per_season = max(count)) %>%
    filter(max_tests_per_season >= threshold_encounters) %>%
    summarize(healthcare_seeking = 1, .groups = 'keep')
  
  assign("aux_df", healthcare_seeking_df)
  healthcare_seeking_df = aux_df %>%
    left_join(y = df_counts, by = "patient_id") %>%
    replace(is.na(.),0) %>%
    mutate(healthcare_seeking = factor(healthcare_seeking)) %>%
    select(-c("calendar_week_number", "calendar_year", "season_of_test",
              "actual_season"))
  
  return (healthcare_seeking_df)
}



influenza_vaccine_df_buildup_RSVburden = function(predictors_df,
                                                  dedup_procedure_df,
                                                  dedup_medication_drug_df, 
                                                  time_range = c(0,-365)){
  # 
  # This function is specific to the RSV burden project. It relies on this project's schema
  # 
  #
  # Description: This function identifies patients who have received the influenza vaccine based on 
  # procedure and medication codes. It takes into account the predictors_df data frame, 
  # dedup_procedure_df, dedup_medication_drug_df, and a time range for considering the vaccination (default: -365 days). 
  # It creates the predictors_df_influenza_vaccine data frame that includes the original predictors along with an 
  # influenza_vaccine column indicating whether the patient received the influenza vaccine or not: 
  # 
  # Inputs:
  # - predictors_df: The input data frame containing predictors or features. It needs to have the following columns:
  #   'patient_id', 'index_date', 'RSV_test_result' (optional)
  # - dedup_procedure_df: The data frame containing deduplicated procedure information, including columns like "patient_id," "code," "date," and "code_system."
  # - dedup_medication_drug_df: The data frame containing deduplicated medication/drug information, including columns like "patient_id," "code," "start_date," and "code_system."
  # - time_range: The time range (in days) to consider for the vaccination (default: c(0,-365)). This means the patients are considered vaccinated if this happens 365 days PRIOR to vaccinaaction
  #
  # Outputs:
  # - predictors_df_influenza_vaccine data frame, which includes the original predictors along with an influenza_vaccine column indicating whether the patient received the influenza vaccine (1) or not (0).
  
  # 4.2 Influenza vaccination
  
  aux_procedure_influenza = dedup_procedure_df %>%
    mutate(sub_code_5 = substring(code, 1,5)) %>%
    mutate(sub_code_6 = substring(code, 1,6)) %>%
    mutate(date = as.Date(date, format = "%Y-%m-%d"))%>%
    filter(code_system == "CPT")%>%
    filter(code %in% cpt_codes_influenza)
  
  influenza_vaccine_cpt <- predictors_df %>% 
    inner_join(y = aux_procedure_influenza, by ='patient_id', keep = FALSE, multiple = "all") %>%
    mutate(time_difference = difftime(date, index_date, units = "days" )) %>% # date_vaccine - index_date 
    filter( (time_difference <= time_range[1]) & (time_difference >= time_range[2])) %>%
    group_by(patient_id, index_date) %>%
    summarize(influenza_vaccine = 1, .groups = 'keep') 
  
  # Second, identify patients with influenza vaccine codes in the medication table
  aux_medication_drug_influenza = dedup_medication_drug_df %>%
    mutate(sub_code_6 = substring(code, 1,6)) %>%
    mutate(sub_code_7 = substring(code, 1,7)) %>%
    mutate(date = as.Date(start_date, format = "%Y-%m-%d"))%>%
    filter(code_system == "RxNorm") %>%
    filter(code %in% rx_codes_influenza)
  
  influenza_vaccine_Rx <- predictors_df %>% 
    inner_join(y = aux_medication_drug_influenza, by ='patient_id', keep = FALSE, multiple = 'all') %>%
    mutate(time_difference = difftime(date, index_date, units = "days" )) %>% # date_vaccine - RSV_test date 
    filter( (time_difference <= 0) & (time_difference >= time_range)) %>%
    group_by(patient_id, index_date)  %>%
    summarize(influenza_vaccine = 1, .groups = 'keep') 
  
  # Finally, aggregate both groups of vaccinated patients
  influenza_vaccine_all_tests = influenza_vaccine_cpt %>%
    full_join(influenza_vaccine_Rx, by = c("patient_id", "index_date")) %>% 
    summarize(influenza_vaccine = 1, .groups = 'keep')
  
  predictors_df_influenza_vaccine = predictors_df %>%
    left_join(y = influenza_vaccine_all_tests, by = c('patient_id','index_date')) %>%
    replace(is.na(.),0) %>% 
    mutate(influenza_vaccine = factor(influenza_vaccine))
  
  return(predictors_df_influenza_vaccine)
}

# Function to check unique patient id count
check_patient_id_count <- function(df, expected_count){
  
  # Specific to RSV burden project
  # 
  # Checks if the number of patients in a predictors dataframe is the correct,
  df_name = deparse(substitute(df))
  
  actual_count <- length(unique(df$patient_id))
  print(paste0('Dataframe ',df_name,' has ',actual_count,' patients'))
  # stopifnot(actual_count == expected_count)
}

# Function to check number of records (rows)
check_row_count <- function(df, expected_count){
  
  # Specific to RSV burden project
  # 
  # Checks if the number of rcords in a predictors dataframe is the correct,
  df_name = deparse(substitute(df))
  actual_count <- nrow(df)
  
  print(paste0('Dataframe ',df_name,' has ',actual_count,' rows'))
  # stopifnot(actual_count == expected_count)
}


