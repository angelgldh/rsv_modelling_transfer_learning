# Predictors are grouped in 6 classes
# 1. Demographic factors
# 2. Flu-like symptoms 
# 3. Comorbidities
# 4. Seasonality
# 5. Healthcare_seeking_behaviour and influenza_vaccine
# 6. Extra predictors

# 0. Baseline data frame ####

predictors_df_phase1 = data.frame(patient_id = lab_result_RSV_df_phase1 $ patient_id,
                           RSV_test_date = lab_result_RSV_df_phase1 $ date,
                           RSV_test_result = lab_result_RSV_df_phase1 $ lab_result_text_val )
predictors_df_phase1 = predictors_df_phase1 %>%
  distinct() %>%
  rename(index_date = RSV_test_date)

# 1. Demographic factors ####

predictors_df_dem_df_phase1 = demographics_df_build_up_RSVburden(predictors_df = predictors_df_phase1,
                                                                 patient_demographic_df = patient_demographic_df_phase1)

summary(predictors_df_dem_df_phase1)


# 2. Flu-like_symptoms ####

flu_like_symptoms_df_phase1 = flu_like_symptoms_df_build_up_RSVburden(predictors_df = predictors_df_phase1, 
                                                                      dedup_diagnosis_df = dedup_diagnosis_df_phase1,
                                                                      window_flu_symptoms = 7)
summary(flu_like_symptoms_df_phase1)

# 3. Comorbidities ####

predictors_cs_df_phase1= comorbidities_df_build_up_RSVburden (predictors_df = predictors_df_phase1, 
                                                              dedup_diagnosis_df = dedup_diagnosis_df_phase1, 
                                                              dedup_procedure_df = dedup_procedure_df_phase1)

summary(predictors_cs_df_phase1)


# 4. Seasonality ####

predictors_season_df_phase1 = seasonality_df_buildup_RSVburden(predictors_df = predictors_df_phase1)
summary(predictors_season_df_phase1)


# 5. Influenza vaccine and healthcare_seeking_behaviour ####

predictors_hc_seekers_df_phase1 = healthcare_seeking_df_buildup_RSVburden(predictors_df = predictors_df_phase1,
                                                                          threshold_encounters = 2)
summary(predictors_hc_seekers_df_phase1)

predictors_influenza_vaccine_df_phase1 = influenza_vaccine_df_buildup_RSVburden(predictors_df = predictors_df_phase1,
                                                                                dedup_procedure_df = dedup_procedure_df_phase1,
                                                                                dedup_medication_drug_df = dedup_medication_drug_df_phase1)
summary(predictors_influenza_vaccine_df_phase1)


# 6. Immunodeficiencies: any_immunodeficiency and n_immunodeficiencies ####

predictors_immuno_df_phase1 = immunodeficiency_df_build_up_RSVburden(predictors_df = predictors_df_phase1,
                                                                     dedup_diagnosis_df = dedup_diagnosis_df_phase1)
dim(predictors_immuno_df_phase1)
summary(predictors_immuno_df_phase1)

# 7. Tumor predictors : ####
predictors_tumor_df_phase1 = tumor_df_build_up_RSVburden(predictors_df = predictors_df_phase1,
                                                         dedup_tumor_df = dedup_tumor_df_phase1)
dim(predictors_tumor_df_phase1)
summary(predictors_tumor_df_phase1)

# Merge them all intermediate dataframes prior to extra predictors ####

length(unique(predictors_df_phase1 $ patient_id)) #48040
length(unique(predictors_df_dem_df_phase1$ patient_id)) #48040
length(unique(flu_like_symptoms_df_phase1 $ patient_id)) #48040
length(unique(predictors_cs_df_phase1 $ patient_id)) #48040
length(unique(predictors_season_df_phase1 $ patient_id)) #48040
length(unique(predictors_influenza_vaccine_df_phase1 $ patient_id)) #48040
length(unique(predictors_hc_seekers_df_phase1 $ patient_id)) #48040
length(unique(predictors_immuno_df_phase1 $ patient_id)) #48040

# Checks: number of records
(dim(predictors_df_phase1)) #87132     
(dim(predictors_df_dem_df_phase1)) #86659      
# demographics has less records as it excludes those tests taken when being outside of the age target!!
(dim(flu_like_symptoms_df_phase1)) #87132    
(dim(predictors_cs_df_phase1)) #87132
(dim(predictors_season_df_phase1)) #87132
((dim(predictors_influenza_vaccine_df_phase1)))#87132
(dim(predictors_hc_seekers_df_phase1))#87132
dim(predictors_immuno_df_phase1) #87132

predictors_df_phase1 = predictors_df_dem_df_phase1 %>%
  left_join(y = flu_like_symptoms_df_phase1, by = c('patient_id','RSV_test_result','index_date'), keep = FALSE) %>%
  left_join(y = predictors_cs_df_phase1, by = c('patient_id','RSV_test_result','index_date'), keep = FALSE)  %>%
  left_join(y = predictors_season_df_phase1, by = c('patient_id','RSV_test_result','index_date'), keep = FALSE) %>%
  left_join(y = predictors_hc_seekers_df_phase1, by = c('patient_id','RSV_test_result','index_date'), keep = FALSE) %>%
  left_join(y = predictors_influenza_vaccine_df_phase1, by = c('patient_id','RSV_test_result','index_date'), keep = FALSE)%>%
  left_join(y = predictors_immuno_df_phase1, by = c('patient_id','RSV_test_result','index_date'), keep = FALSE) %>%
  left_join(y = predictors_tumor_df_phase1, by = c('patient_id','RSV_test_result','index_date'), keep = FALSE) 


# remove non-common characters to avoid potential inconsistencies
names(predictors_df_phase1) = gsub("-", "", names(predictors_df_phase1))
names(predictors_df_phase1) = gsub(" ", "_", names(predictors_df_phase1))
names(predictors_df_phase1) = gsub("\\(", "", names(predictors_df_phase1))
names(predictors_df_phase1) = gsub("\\)", "", names(predictors_df_phase1))
names(predictors_df_phase1) = gsub(",", "", names(predictors_df_phase1))
names(predictors_df_phase1) = gsub("\\.", "", names(predictors_df_phase1))
names(predictors_df_phase1)

dim(predictors_df_phase1)
summary(predictors_df_phase1)

# 6. Continue to work on extra predictors: n_encounters, key_comorbidities, prev_positive_rsv, tendency_to_positivity, previous_test_daydiff, n_tests_that_day ####

extra_predictors_df_phase1 = predictors_df_phase1

## 6.1. n_symptoms #### 
flu_like_symptoms = names(extra_predictors_df_phase1)[9:23]
extra_predictors_df_phase1 = create_n_symptoms(extra_predictors_df_phase1, flu_like_symptoms)

dim(extra_predictors_df_phase1)

## 6.2. n_encounters ####
extra_predictors_df_phase1 = create_n_encounters(extra_predictors_df_phase1)

dim(extra_predictors_df_phase1)


## 6.3. key_comorbidities ####
extra_predictors_df_phase1 = create_key_comorbidities(extra_predictors_df_phase1)
dim(extra_predictors_df_phase1)


## 6.4. prev_positive_rsv ####

extra_predictors_df_phase1 = detect_prev_positive_rsv_tests(extra_predictors_df_phase1) 

dim(extra_predictors_df_phase1)

## 6.5. tendency_to_positivity ####

extra_predictors_df_phase1 = detect_tendency_to_positivity(extra_predictors_df_phase1)

dim(extra_predictors_df_phase1)

## 6.6. previous_test_daydiff ####

extra_predictors_df_phase1 = detect_previous_test_day_diff(extra_predictors_df_phase1)

dim(extra_predictors_df_phase1)

## 6.7. n_tests_that_day ####

extra_predictors_df_phase1 = detect_n_tests_that_day(extra_predictors_df_phase1)
dim(extra_predictors_df_phase1)
names(extra_predictors_df_phase1)


# Merge all back ####
extra_predictors_df_phase1 = extra_predictors_df_phase1 %>%
  select(patient_id, index_date, RSV_test_result,n_symptoms, n_encounters, key_comorbidities,
         prev_positive_rsv, tendency_to_positivity, previous_test_daydiff, n_tests_that_day)

summary(extra_predictors_df_phase1)

# Final step: merge all and load to scratch space ####

rsv_predictors_df_phase1 = predictors_df_phase1 %>%
  left_join(y = extra_predictors_df_phase1, by = c('patient_id', 'index_date', 'RSV_test_result'), keep = FALSE)

dim(rsv_predictors_df_phase1)
summary(rsv_predictors_df_phase1)


db <- RocheData::get_data(data = "trinetx_rsv")

scratch_space = "scr_scr_449_rwddia434_rsv"
db$schema = scratch_space
db$connect()
db$is_connected()
conn = db$con

dbWriteTable(conn, "rsv_predictors_df_phase1", rsv_predictors_df_phase1)

db $ disconnect()

