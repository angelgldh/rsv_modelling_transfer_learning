def make_it_categorical(df, is_phase1 = True):

  factor_columns = ["sex","race","marital_status","patient_regional_location","age_group",
                     "Conjuctivitis", "Acute_upper_respiratory_infection", "Influenza","Pneumonia","Bronchitis"
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
                     ,"Peripheral_Vascular","CVD","COPD","Dementia","Paralysis","Diabetes","Diabetes_complications"
                     ,"Renal_disease"
                     ,"mild_liver_disease"
                     ,"moderate_liver_disease"
                     ,"Peptic_Ulcer_Disease"
                     ,"rheuma_disease"
                     ,"AIDS"
                     ,"Asthma_chronic"
                     ,"calendar_year"
                     ,"healthcare_seeking"
                     ,"influenza_vaccine",
                     "key_comorbidities",
                     "n_tests_that_day"
                     ,"any_immunodeficiency"
                     , 'tumor_indicator'
                     ,'tumor_last_year'
                     ,'is_metastatic']
  if is_phase1:
    factor_columns.append('RSV_test_result')

  # Option A: for loop
  # for column_name in factor_columns:
  #   if column_name in df.columns:
  #     df[column_name] = df[column_name].astype('categorical')

  # Option B: slightly optimized
  factor_columns_set = set(factor_columns)
  columns_to_categorize = list(factor_columns_set.intersection(set(df.columns)))

  df[columns_to_categorize] = df[columns_to_categorize].apply(lambda x : x.astype('category'))

  return df



def summary_function_rsv(df, is_phase1 = True):

  factor_columns = ["sex","race","marital_status","patient_regional_location","age_group",
                     "Conjuctivitis", "Acute_upper_respiratory_infection", "Influenza","Pneumonia","Bronchitis"
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
                     ,"Peripheral_Vascular","CVD","COPD","Dementia","Paralysis","Diabetes","Diabetes_complications"
                     ,"Renal_disease"
                     ,"mild_liver_disease"
                     ,"moderate_liver_disease"
                     ,"Peptic_Ulcer_Disease"
                     ,"rheuma_disease"
                     ,"AIDS"
                     ,"Asthma_chronic"
                     ,"calendar_year"
                     ,"healthcare_seeking"
                     ,"influenza_vaccine",
                     "key_comorbidities",
                     "n_tests_that_day"
                     ,"any_immunodeficiency"
                     , 'tumor_indicator'
                     ,'tumor_last_year'
                     ,'is_metastatic']

  if is_phase1:
    factor_columns.append('RSV_test_result')

  non_factor_columns = [column_name for column_name in df.columns if column_name not in factor_columns]

  print(df[non_factor_columns].describe())
  print('\n')

  for col in factor_columns:
    print(f'Column: {col}')
    print(df[col].value_counts())
    print('\n')



