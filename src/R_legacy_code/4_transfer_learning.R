# Transfer learning to phase 2 data #


# 1. Initial approach: re-running selected model ####

target_model_from_phase1 = trained_models_ff4 $ ff4_rf_down_8020
target_ff = ff4
selected_features = strsplit( as.character(target_ff)[3], "\\s\\+\\s")[[1]]

validation_data_phase2 = rsv_predictors_df_phase2 %>% select(all_of(selected_features))

predictions_probs = predict(target_model_from_phase1, 
                            validation_data_phase2, type = "prob")[, "Positive"]

target_thre = (all_model_metrics_df%>% filter(method == 'rf', sampling == 'down'))$optimal_threshold
prediction_labels = ifelse(predictions_probs >= 0.5, "Positive", "Negative")

# 2. More advanced transfer learning techniques ####

# 3. Calculate indidence in new population ####