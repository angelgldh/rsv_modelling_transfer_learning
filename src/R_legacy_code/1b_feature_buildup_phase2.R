# Build_up of covariates for classification model
# Follows same structures as phase 1 feature build-up
# Has a main twist:
# - At phase 1 > included every encounter encoding for RSV test
# - At phase 2 > included every encounter encoding for flu-like symptom or test


# Covariates can be segmented according to 
# 1. Demographics
# 2.1 Flu-like symptoms presence
# 2.2. Comorbidities
# 3. Seasonality

# Baseline data frame: built from diagnosis codes lab_test_codes
(n_char_dx_codes = unique(nchar(codes_dx_flu_like $ code))) # 3 5
(n_char_labtests_codes = unique(nchar(codes_test_flu_like $Code))) # 6 7

# Extract encounters having flu_like diagnoses and encounters
flu_like_dx_encounter_df = dedup_diagnosis_phase2_df %>% 
  select(patient_id, encounter_id, code, date)%>% 
  mutate(sub_code_3 = substr(code, 1,3)) %>%
  mutate(sub_code_5 = substr(code, 1,5)) %>%
  filter((sub_code_3 %in% codes_dx_flu_like $ code) | (sub_code_5 %in% codes_dx_flu_like $ code)) %>% 
  distinct(patient_id, encounter_id, date)

flu_like_tests_encounter_df = dedup_lab_result_phase2_df %>%
  select(patient_id, encounter_id, code, date) %>%
  mutate(sub_code_6 = substr(code, 1,6)) %>%
  mutate(sub_code_7 = substr(code, 1,7)) %>%
  filter((sub_code_6 %in% codes_test_flu_like $ Code) | (sub_code_7 %in% codes_test_flu_like $ Code)) %>% 
  distinct(patient_id, encounter_id, date)


flu_like_encounters_phase2_df = bind_rows(flu_like_dx_encounter_df, flu_like_tests_encounter_df) %>% 
  distinct(patient_id, date)

# Baseline data frame

predictors_phase2_df = data.frame(patient_id =  flu_like_encounters_phase2_df $ patient_id,
                                  index_date = flu_like_encounters_phase2_df $date ) %>%
  mutate(index_date = as.Date(index_date, format = '%Y-%m-%d'))
dim(predictors_phase2_df); summary(predictors_phase2_df)



# 1. Demographics predictors ####
# Analogously to phase 1, build-up the demographic variables

predictors_df_dem_phase2_df = demographics_df_build_up_RSVburden(predictors_df = predictors_phase2_df,
                                                                 patient_demographic_df = patient_demographic_phase2_df)

summary(predictors_df_dem_phase2_df)


# 2. Flu-like symptoms ####

flu_like_symptoms_phase2_df = flu_like_symptoms_df_build_up_RSVburden(predictors_df = predictors_phase2_df, 
                                                                      dedup_diagnosis_df = dedup_diagnosis_phase2_df,
                                                                      window_flu_symptoms = 7)
summary(flu_like_symptoms_phase2_df)

# 3. Comorbidities ####

predictors_cs_phase2_df= comorbidities_df_build_up_RSVburden (predictors_df = predictors_phase2_df, 
                                                              dedup_diagnosis_df = dedup_diagnosis_phase2_df, 
                                                              dedup_procedure_df = dedup_procedure_phase2_df)

summary(predictors_cs_phase2_df)


# 4. Seasonality #### 

predictors_season_phase2_df = seasonality_df_buildup_RSVburden(predictors_df = predictors_phase2_df)
summary(predictors_season_phase2_df)

# 5. Additional predictors: healthcare_seeking_behaviour and influenza_vaccine" ####

predictors_hc_seekers_phase2_df = healthcare_seeking_df_buildup_RSVburden(predictors_df = predictors_phase2_df,
                                                                          threshold_claims = 5)
summary(predictors_hc_seekers_phase2_df)

predictors_influenza_vaccine_phase2_df = influenza_vaccine_df_buildup_RSVburden(predictors_df = predictors_phase2_df,
                                                                                dedup_procedure_df = dedup_procedure_phase2_df,
                                                                                dedup_medication_drug_df = dedup_medication_drug_phase2_df)
summary(predictors_influenza_vaccine_phase2_df)


# 6. Immunodeficiencies ####
predictors_immuno_phase2_df = immunodeficiency_df_build_up_RSVburden(predictors_df = predictors_phase2_df,
                                                                     dedup_diagnosis_df = dedup_diagnosis_phase2_df)
dim(predictors_immuno_phase2_df)
summary(predictors_immuno_phase2_df)

# 7. Tumor predictors : ####
predictors_tumor_phase2_df = tumor_df_build_up_RSVburden(predictors_df = predictors_phase2_df,
                                                         dedup_tumor_df = dedup_tumor_phase2_df)
dim(predictors_tumor_phase2_df)
summary(predictors_tumor_phase2_df)


# Final step > Merge all together ####

data_frames <- list(predictors_phase2_df, predictors_df_dem_phase2_df, flu_like_symptoms_phase2_df, 
                    predictors_cs_phase2_df, predictors_season_phase2_df, predictors_hc_seekers_phase2_df,
                    predictors_influenza_vaccine_phase2_df,
                    predictors_immuno_phase2_df, predictors_tumor_phase2_df)

expected_patient_count = 60000
expected_row_count = 551239 
# the demographics df is taken as reference because it is the one with the more restrictive 
# exclusion criterion: it removes all records of patients younger than 18 years old

for (df in data_frames){
  check_patient_id_count(df, expected_patient_count)
  check_row_count(df, expected_row_count)
  
}



rsv_predictors_df_phase2 = predictors_df_dem_phase2_df %>%
  left_join(y = flu_like_symptoms_phase2_df, by = c('patient_id','index_date'), keep = FALSE) %>%
  left_join(y = predictors_cs_phase2_df, by = c('patient_id','index_date'), keep = FALSE)  %>%
  left_join(y = predictors_season_phase2_df, by = c('patient_id','index_date'), keep = FALSE) %>%
  left_join(y = predictors_hc_seekers_phase2_df, by = c('patient_id','index_date'), keep = FALSE) %>%
  left_join(y = predictors_influenza_vaccine_phase2_df, by = c('patient_id','index_date'), keep = FALSE) %>%
  left_join(y = predictors_immuno_phase2_df, by = c('patient_id','index_date'), keep = FALSE)%>%
  left_join(y = predictors_tumor_phase2_df, by = c('patient_id','index_date'), keep = FALSE)


dim(rsv_predictors_df_phase2)
summary(rsv_predictors_df_phase2)

# remove non-common characters to avoid potential inconsistencies
names(rsv_predictors_df_phase2) = gsub("-", "", names(rsv_predictors_df_phase2))
names(rsv_predictors_df_phase2) = gsub(" ", "_", names(rsv_predictors_df_phase2))
names(rsv_predictors_df_phase2) = gsub("\\(", "", names(rsv_predictors_df_phase2))
names(rsv_predictors_df_phase2) = gsub(")", "", names(rsv_predictors_df_phase2))
names(rsv_predictors_df_phase2) = gsub(",", "", names(rsv_predictors_df_phase2))
names(rsv_predictors_df_phase2) = gsub("_\\.", "_", names(rsv_predictors_df_phase2))
names(rsv_predictors_df_phase2)


# 5. Extra predictors: n_Claims, key_comorbidities, prev_positive_rsv, tendency_to_positivity, previous_test_daydiff, n_tests_that_day ####

extra_predictors_phase2_df = rsv_predictors_df_phase2

## 5.1. n_symptoms #### 

flu_like_symptoms = names(extra_predictors_phase2_df)[8:22]
extra_predictors_phase2_df = create_n_symptoms(extra_predictors_phase2_df, flu_like_symptoms)

dim(extra_predictors_phase2_df)

## 5.2. n_claims ####

extra_predictors_phase2_df = create_n_encounters(extra_predictors_phase2_df)

dim(extra_predictors_phase2_df)


## 5.3. key_comorbidities ####

extra_predictors_phase2_df = create_key_comorbidities(extra_predictors_phase2_df)
dim(extra_predictors_phase2_df)
summary(extra_predictors_phase2_df)


## 5.4. prev_positive_rsv ####
# This one needs to be defined alternatively to phase 1, as not every record
# is a RSV lab test

previous_tests_rsv_df = detect_prev_positive_rsv_tests(rsv_tests_phase2_df) %>%
  rename(index_date = RSV_test_date) %>%
  select(-c(RSV_test_result))%>%
  group_by(patient_id, index_date) %>%
  summarise(prev_positive_rsv = max(prev_positive_rsv), .groups = 'keep')

extra_predictors_phase2_df = extra_predictors_phase2_df %>%
  left_join(y = previous_tests_rsv_df, by = c('patient_id','index_date')) %>% 
  arrange(patient_id, index_date) %>%
  group_by(patient_id)%>%
  fill(prev_positive_rsv, .direction = "down") %>%
  mutate(prev_positive_rsv = replace_na(prev_positive_rsv, 0)) %>%
  ungroup()

dim(extra_predictors_phase2_df)
summary(extra_predictors_phase2_df)


## 5.5. tendency_to_positivity ####

tendency_to_positivity_df = detect_tendency_to_positivity(rsv_tests_phase2_df) %>%
  select(patient_id, tendency_to_positivity) %>%
  distinct()

extra_predictors_phase2_df = extra_predictors_phase2_df %>%
  left_join(y = tendency_to_positivity_df, by = 'patient_id') %>%
  mutate(tendency_to_positivity = replace_na(tendency_to_positivity, 0))

dim(extra_predictors_phase2_df)
summary(extra_predictors_phase2_df)

## 5.6. previous_test_daydiff ####

previous_test_daydiff_df = detect_previous_test_day_diff(rsv_tests_phase2_df) %>%
  select(-c(RSV_test_result))%>%
  rename(index_date = RSV_test_date) %>%
  group_by(patient_id, index_date) %>%
  summarise(min(previous_test_daydiff), .groups = 'keep') %>%
  rename(previous_test_daydiff = `min(previous_test_daydiff)`)

extra_predictors_phase2_df = extra_predictors_phase2_df %>%
  left_join(y = previous_test_daydiff_df, by = c('patient_id', 'index_date')) %>%
  mutate(previous_test_daydiff = replace_na(previous_test_daydiff, 365*3))

dim(extra_predictors_phase2_df)


## 5.7. n_tests_that_day ####

n_tests_that_day_df = detect_n_tests_that_day(rsv_tests_phase2_df) %>%
  select(-c(RSV_test_result))%>%
  rename(index_date = RSV_test_date) %>%
  distinct()

extra_predictors_phase2_df = extra_predictors_phase2_df %>%
  left_join(y = n_tests_that_day_df, by = c('patient_id', 'index_date')) %>%
  mutate(n_tests_that_day = replace_na(n_tests_that_day, as.factor(0)))

dim(extra_predictors_phase2_df)
summary(extra_predictors_phase2_df)

## Merge_all extra features ##

extra_predictors_phase2_df = extra_predictors_phase2_df %>%
  select(patient_id, index_date, n_symptoms, n_encounters, key_comorbidities,
         prev_positive_rsv, tendency_to_positivity, previous_test_daydiff, n_tests_that_day)

dim(rsv_predictors_df_phase2)

rsv_predictors_df_phase2 = rsv_predictors_df_phase2 %>%
  left_join(y = extra_predictors_phase2_df, by = c('patient_id', 'index_date'), keep = FALSE)

dim(rsv_predictors_df_phase2)

## Final check: missingness ##
missing_per_column =  lapply(rsv_predictors_df_phase2, function(x) sum(is.na(x)))

missing_per_column[names(missing_per_column[missing_per_column > 0])]

rsv_predictors_df_phase2 = rsv_predictors_df_phase2 %>%
  filter(!is.na(sex))

# Load the data to scratch space ####

db <- RocheData::get_data(data = "trinetx_rsv")

scratch_space = "scr_scr_449_rwddia434_rsv"

db$schema = scratch_space
db$connect()
db$is_connected()
conn = db$con
dbWriteTable(conn, "rsv_predictors_df_phase2", rsv_predictors_df_phase2)
db $ disconnect()

