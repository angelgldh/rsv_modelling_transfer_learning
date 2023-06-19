# Script to evaluate the performance of models developed in the phase 1 stage 
# 1. Performance metrics
# 2. Feature importance
# 4. Clustering of models according to importance
# 5. Explainable AI: SHAP and LIME


# !!! User input: ####
# Indicate the whole path to the database storing the trained models

path_names = c("~/R/models_store/ff4_models_8020_20230530_rf.db",
             "~/R/models_store/ff4_models_8020_20230530_gbm.db")

trained_models_all <- list()
for (name in path_names){
  trained_models = load_models_from_custom_database(name)
  
  trained_models_all = c(trained_models_all, trained_models)
}


# 1. Performance metrics ####
# These metrics include:
# - F1 score
# - Sensitivity/recall
# - Specificity
# - PPV / precision
# - NPV
# - Area under the curve

# Make sure the following are teh same as in 2_ML_modelling.R
# # Indicate training and validation sets
# training_set = trainData_phase1
# validation_set = testData_phase1

# 1.1 AUC and confusion matric ####
# Area under the curve (AUC)
model_list_roc <- trained_models_all %>% map(test_roc, validation_set = validation_set)
obtain_auc(model_list_roc)

# Confusion matrix and performance metrics
model_list_cm = trained_models_all %>% map(test_cm, validation_data = validation_set)
model_list_cm_thres06 = trained_models_all %>% map(test_cm, validation_data = validation_set,
                                                   threshold_pred = 0.6)
for (ii in 1:length(trained_models_all)){
  print(names(trained_models_all)[ii])
  print(model_list_cm[[ii]] $ byClass)
}



# 1.2 Performance metrics: optimal threshold per model ####

all_model_metrics_df = metrics_model_list(model_list = trained_models_all, 
                                          validation_df = validation_set)

# 1.3 Confidence intervals for all metrics > bootstrap ####

cis_pROC_df = data.frame(auc_lower = numeric(), auc_median = numeric(), auc_upper = numeric(),
                         sensitivity_lower = numeric(), sensitivity_median = numeric(), sensitivity_upper = numeric(),
                         specificity_lower = numeric(), specificity_median = numeric(), specificity_upper = numeric())

for (ii in 1:nrow(all_model_metrics_df)){
  print(names(trained_models_all)[ii])
  metrics.ii = all_model_metrics_df[ii,]
  roc.ii = model_list_roc[[ii]]
  
  n_boot = 500
  
  # AUC CI
  ci_auc = ci.auc(roc.ii, boot.n = n_boot, conf.level = 0.95)
  
  # Sensitivity CI
  ci_se = ci.se(roc.ii, specificities = as.numeric(metrics.ii $ specificity), boot.n = n_boot, conf.level = 0.95)
  
  # Specificity CI
  ci_sp = ci.sp(roc.ii, sensitivities = as.numeric(metrics.ii $ sensitivity), boot.n = n_boot, conf.level = 0.95)
  
  cis_pROC_df[ii,] = c(ci_auc, ci_se, ci_sp)
  
}
rownames(cis_pROC_df) <- names(trained_models_all)



# 2. Feature importance ####
# Indicate the model to be evaluated 
rf_target_model = trained_models_all $ ff4_rf_down_8020

rf_model_imp = varImp(rf_target_model, scale = TRUE)
n_top_features = 15
n_top_features = length(rf_target_model $ coefnames)

p1 = rf_model_imp$importance%>%
  as.data.frame()%>%
  rownames_to_column() %>%
  ggplot(aes(x = reorder(rowname, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "#1F77B4", alpha = 0.8) +
  coord_flip()  
p1

