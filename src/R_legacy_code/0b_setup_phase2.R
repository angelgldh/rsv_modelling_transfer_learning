# All needed libraries have been loaded as part of phase 1


# 1. Connect to trinetx_phase2 and load the data ####

db <- RocheData::get_data(data = "trinetx_rsv")

schema_phase1 = "trinetx_rsv_sdm_2022_08"
schema_phase2 = "trinetx_rsv_sdm_2023_04"
scratch_space = "scr_scr_449_rwddia434_rsv"

db$schema <-schema_phase2

if (db $ schema == schema_phase2){
  
  db$connect()
  
  print('Schema is CORRECT! Loading tables ...')
  
  cohort_details_phase2 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "cohort_details")) %>% collect()
  
  dataset_details_phase2 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "dataset_details")) %>% collect()
  
  patient_cohort_phase2_df <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "patient_cohort")) %>% collect()
  
  patient_demographic_phase2_df <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "patient_demographic")) %>% collect()
  
  print('Loaded basic demographics tables!')
  print('Now loading diagnosis, procedure, medication_drug and lab_result... This may take up to ~90 mins')
  
  diagnosis_phase2_df <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "diagnosis")) %>% collect()
  
  procedure_phase2_df <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "procedure")) %>% collect()
  
  medication_drug_phase2_df <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "drug")) %>% collect()
  
  lab_result_phase2_df<- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "lab_result")) %>% collect()
  
  encounter_phase2_df<- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "encounter")) %>% collect()
  
  tumor_phase2_df<- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "tumor")) %>% collect()
  
  print('All needed tables are loaded!')
  db$disconnect()
  
}else{
  print("Schema is not correct, double check!")
}

# 2. Load all needed codes, following same steps as in phase 1 ####

load_codes_RSVburden()


# 3. Compute and clean lab_result_RSV_df dataframe ####
lab_result_RSV_phase2_df = lab_result_phase2_df %>% 
  filter(code %in% rsv_codes) %>%
  filter(!is.na(lab_result_text_val), lab_result_text_val != "Unknown") %>%
  mutate(lab_result_text_val = factor(lab_result_text_val)) %>%
  mutate(code_system = factor(code_system))

rsv_tests_phase2_df = lab_result_RSV_phase2_df %>%
  mutate(
    RSV_test_date = as.Date(date, format = '%Y-%m-%d'),
    RSV_test_result = lab_result_text_val) %>%
  select(patient_id, RSV_test_date, RSV_test_result)

summary(rsv_tests_phase2_df)

# RSV positive patients
RSV_patients_list = extract_positive_negative_all_patients_RSVburden(lab_result_RSV_phase2_df)
length(RSV_patients_list $RSV_negative_patients)

# 4. Tables dedup ####
target_df = c('diagnosis','procedure', 'lab_result', 'medication_drug','encounter')

dedup_dfs <- lapply(target_df, function(t) {
  get(paste0(t, '_phase2_df')) %>% distinct()
})

names(dedup_dfs) <- paste0('dedup_', (target_df), '_phase2_df')
list2env(dedup_dfs, envir = .GlobalEnv)


# 5. Check if all patients are present where they should be ####

pat_ids_dem = unique(patient_demographic_phase2_df $ patient_id)
pat_ids_dx = unique(dedup_diagnosis_phase2_df $ patient_id)
pat_ids_lab = unique(dedup_lab_result_phase2_df $ patient_id)
pat_ids_proc =unique(dedup_procedure_phase2_df $ patient_id)
pat_ids_rsv_tests = unique(rsv_tests_phase2_df $ patient_id)
length(pat_ids_rsv_tests)
# pat_ids_enc =unique(encounter_phase2_df $ patient_id)

sum(pat_ids_dx %in% pat_ids_dem)
sum(pat_ids_dx %in% pat_ids_lab)
sum(pat_ids_dx %in% pat_ids_proc)
sum(pat_ids_dx %in% pat_ids_rsv_tests)
# sum(pat_ids_dx %in% pat_ids_enc)

sum(pat_ids_lab %in% pat_ids_dem)
sum(pat_ids_lab %in% pat_ids_proc)
sum(pat_ids_lab %in% pat_ids_rsv_tests)
# sum(pat_ids_lab %in% pat_ids_enc)

sum(pat_ids_proc %in% pat_ids_dem)
sum(pat_ids_proc %in% pat_ids_rsv_tests)
# sum(pat_ids_proc %in% pat_ids_enc)

# sum(pat_ids_enc %in% pat_ids_dem)


# 6. If needed, load directly the predictors data from phase 2 ####

rsv_predictors_df_phase2 = load_phase2_predictors_df()
summary(rsv_predictors_df_phase2)




