# ML Modelling:
# 0. Loading of modelling data (from phase 1)
# 1. Definition of models and ML techniques 
# 2. Main training loop

# 0. Loading of modelling data (phase 1 data) ####

data_phase_1_is_loaded = T

if (data_phase_1_is_loaded){
  print('rsv_predictors_df_phase1 is laoded already!')
}else{
  rsv_predictors_df_phase1 = load_phase1_predictors_df()
}
dim(rsv_predictors_df_phase1)
names(rsv_predictors_df_phase1)
summary(rsv_predictors_df_phase1)

# Split training and test data
set.seed(123) # Important !!! keep seed at 123
df = rsv_predictors_df_phase1
training_factor = 0.8
trainIndex <- caret::createDataPartition(df $ RSV_test_result, p = training_factor, list = FALSE, times = 1)
trainData_phase1  <- df[ trainIndex,]
testData_phase1  <- df[-trainIndex,] 

summary(testData_phase1 $ RSV_test_result) # number of Positive results should be 530 !!!


# 1. Models, ML techniques and sampling methods ####
# 1.1. Models ####

# Baseline model
ff0 = as.formula('RSV_test_result ~ sex + race + marital_status + patient_regional_location + 
                 age_group + Conjuctivitis + Acute_upper_respiratory_infection + Influenza + 
                 Pneumonia + Bronchitis + Acute_lower_respiratory_infection_other + Rhinitis + 
                 Other_COPD + Asthma + Symptoms_and_signs__circulatory_and_respiratory + 
                 Symptoms_and_signs__digestive_system_and_abdomen + 
                 Symptoms_and_signs__skin_and_subcutaneous_tissue + 
                 Symptoms_and_signs__cognition_perception_emotional_state_and_behaviour + 
                 General_symptoms_and_signs + COVID19_related + any_symptom + 
                 Acute_myocardial_infarction + Hystory_myocardial_infarction + 
                 Congestive_heart_failure + Peripheral_Vascular + CVD + COPD + Dementia +
                 Paralysis + Diabetes + Diabetes_complications + Renal_disease + 
                 mild_liver_disease + moderate_liver_disease + Peptic_Ulcer_Disease + 
                 rheuma_disease + AIDS + Asthma_chronic + CCI + sine + cosine + 
                 calendar_year + healthcare_seeking + influenza_vaccine')

ff2 = as.formula('RSV_test_result ~ sine + cosine + sex + marital_status + age_group + 
    Pneumonia + Influenza + Bronchitis + Asthma + Symptoms_and_signs__circulatory_and_respiratory + 
    Symptoms_and_signs__digestive_system_and_abdomen + General_symptoms_and_signs + 
    any_symptom + Acute_myocardial_infarction + Congestive_heart_failure + 
    Peripheral_Vascular + COPD + Diabetes_complications + rheuma_disease + 
    AIDS + healthcare_seeking + influenza_vaccine + CCI + n_symptoms + 
    n_claims + key_comorbidities + past_positive')

ff3 = as.formula('RSV_test_result ~ sine + cosine + sex + marital_status + age_group + 
    Pneumonia + Influenza + Bronchitis + Asthma + Symptoms_and_signs__circulatory_and_respiratory + 
    Symptoms_and_signs__digestive_system_and_abdomen + General_symptoms_and_signs + 
    any_symptom + Acute_myocardial_infarction + Congestive_heart_failure + 
    Peripheral_Vascular + COPD + Diabetes_complications + rheuma_disease + 
    AIDS + healthcare_seeking + influenza_vaccine + CCI + n_symptoms + 
    n_claims + key_comorbidities + prev_positive_rsv')

ff4 = as.formula('RSV_test_result ~ sine + cosine + sex + marital_status + age_group + 
    Pneumonia + Influenza + Bronchitis + Asthma + Symptoms_and_signs__circulatory_and_respiratory + 
    Symptoms_and_signs__digestive_system_and_abdomen + General_symptoms_and_signs + 
    any_symptom + Acute_myocardial_infarction + Congestive_heart_failure + 
    Peripheral_Vascular + COPD + Diabetes_complications + rheuma_disease + 
    AIDS + healthcare_seeking + influenza_vaccine + CCI + n_symptoms + 
    n_claims + key_comorbidities + prev_positive_rsv + tendency_to_positivity')
ff5 = as.formula('RSV_test_result ~ sine + cosine + sex + marital_status + age_group + 
    Pneumonia + Influenza + Bronchitis + Asthma + Symptoms_and_signs__circulatory_and_respiratory + 
    Symptoms_and_signs__digestive_system_and_abdomen + General_symptoms_and_signs + 
    any_symptom + Acute_myocardial_infarction + Congestive_heart_failure + 
    Peripheral_Vascular + COPD + Diabetes_complications + rheuma_disease + 
    AIDS + healthcare_seeking + influenza_vaccine + CCI + n_symptoms + 
    n_claims + key_comorbidities + prev_positive_rsv + previous_test_daydiff')
ff6 = as.formula('RSV_test_result ~ sine + cosine + sex + marital_status + age_group + 
    Pneumonia + Influenza + Bronchitis + Asthma + Symptoms_and_signs__circulatory_and_respiratory + 
    Symptoms_and_signs__digestive_system_and_abdomen + General_symptoms_and_signs + 
    any_symptom + Acute_myocardial_infarction + Congestive_heart_failure + 
    Peripheral_Vascular + COPD + Diabetes_complications + rheuma_disease + 
    AIDS + healthcare_seeking + influenza_vaccine + CCI + n_symptoms + 
    n_claims + key_comorbidities + prev_positive_rsv + previous_test_daydiff + n_tests_that_day')

# 1.2. ML techniques under study ####
ml_methods = c("LogitBoost", "gbm", "xgbTree", "rf")

# 1.3. Resampling methods ####
resampling_methods = c('original','down','weighted','up','smote')


# 2. Main training loop ####

# USER Input!!!
# - Choose model to evaluate
# - Indicate training and validation sets
# - Split factor
# - Path to the folder where models are to be saved

# Choose the model to evaluate
ff = ff4
ff_character = 'ff4'

# Indicate training and validation sets
training_set = trainData_phase1
validation_set = testData_phase1

# Split factor
if (training_factor == 0.8){
  spl = '8020' # indicate the data split
}else{
  spl = 'unspecified'
}

# Path to the folder where modls are saved
model_folder = "~/R/models_store"
  
today_ch = as.character(today())


for (ii in 1:length(ml_methods) ){
  m = ml_methods[ii]
  
  db_name_2 = paste0(ff_character,"_models_",spl,"_",today_ch,"_",m,".db")
  model_name_0 = paste0(ff_character,"_",m)
  
  
  registerDoParallel(detectCores())
  
  
  train_control <- trainControl(method="cv", number=5, 
                                summaryFunction=twoClassSummary, 
                                classProbs=TRUE, 
                                savePredictions="all",
                                search="grid")
  
  
  
  if ("original" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: none"))
    
    # No rebalance
    orig_fit = caret::train(ff, data=trainData, 
                            method=m,
                            trControl=train_control,
                            preProcess = c("center", "scale"),
                            tuneLength = 5,
                            metric="ROC")  
    
    print_auc(orig_fit, testData =  validation_set)
    
    
    model_name= paste0(model_name_0,"_original_", spl)
    save_model(folder_name = model_folder, db_name = db_name_2,
               model = orig_fit, model_name = model_name)
  }
  
  if ("weighted" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: weigths"))
    
    # weighted classification
    model_weights <- ifelse(training_set$RSV_test_result == "Negative",
                            (1/table(training_set$RSV_test_result)[1]) * 0.5,
                            (1/table(training_set$RSV_test_result)[2]) * 0.5)
    #train_control$seeds <- orig_fit$control$seeds
    weighted_fit = caret::train(ff, data = training_set, method = m,
                                trControl = train_control, metric = "ROC",
                                weights = model_weights,
                                preProcess = c("center", "scale"),
                                tuneLength = 5)
    
    print_auc(weighted_fit, testData =  validation_set)
    
    model_name= paste0(model_name_0,"_weighted_", spl)
    save_model(folder_name = model_folder,db_name = db_name_2,
               model = weighted_fit, model_name = model_name)    
  }
  
  
  if ("down" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: down"))
    
    # Down-sampled model
    train_control$sampling <- "down"
    train_control$seeds <- orig_fit$control$seeds
    
    down_fit <- caret::train(ff, data = training_set,
                             method = m,
                             metric = "ROC", trControl = train_control,
                             preProcess = c("center", "scale"),
                             tuneLength = 5)
    
    print_auc(down_fit, testData =  validation_set)
    
    model_name= paste0(model_name_0,"_down_", spl)
    save_model(folder_name = model_folder, db_name = db_name_2,
               model = down_fit, model_name = model_name)
  }
  
  if ("up" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: up"))
    
    # Up-sampled model
    train_control$sampling <- "up"
    train_control$seeds <- orig_fit$control$seeds
    
    up_fit <- caret::train(ff, data = training_set,  method = m,
                           metric = "ROC",  trControl = train_control,
                           preProcess = c("center", "scale"),
                           tuneLength = 5)
    
    print_auc(up_fit, testData =  validation_set)
    
    model_name= paste0(model_name_0,"_up_", spl)
    save_model(folder_name = model_folder, db_name = db_name_2,
               model = up_fit, model_name = model_name)
  }
  
  if ("smote" %in% resampling_methods){
    print(paste0("Method: ",m," resampling: SMOTE"))
    
    # SMOTE model
    train_control$sampling <- "smote"
    train_control$seeds <- orig_fit$control$seeds
    
    smote_fit <- caret::train(ff, data = training_set,  method = m, verbose = FALSE,
                              metric = "ROC",  trControl = train_control,
                              preProcess = c("center", "scale"),
                              tuneLength = 5)
    
    print_auc(smote_fit, testData =  validation_set)
    
    model_name= paste0(model_name_0,"_smote_", spl)
    save_model(folder_name = model_folder,db_name = db_name_2,
               model = smote_fit, model_name = model_name)
  }
  
  stopImplicitCluster()
  
  
}
