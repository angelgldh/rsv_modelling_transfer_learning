# 0. Load all needed libraries ####

library(tidyverse)
library(glue)
library(RocheData)
library(lubridate)
library(gtsummary)
library(DBI)
library(sqldf)
library(gtsummary)
library(ggplot2)
library(rapportools)
library(dbplyr)
library(dplyr)
library(RSQLite)
library(ggcorrplot)
library(cowplot)
library(imbalance)
library(ROSE)
library(unbalanced)
library(caret)
library(pROC)
library(dummies)
library(ipred)
library(MLmetrics)
library(xgboost)
library(e1071) 
library(doParallel)

setwd("~/R/rwddia_434")
source('scripts/00_aux_functions.R')


# 1. Connect to trinetx_phase2 and load the data ####

db <- RocheData::get_data(data = "trinetx_rsv")

schema_phase1 = "trinetx_rsv_sdm_2022_08"
schema_phase2 = "trinetx_rsv_sdm_2023_04"
scratch_space = "scr_scr_449_rwddia434_rsv"

db$schema <-schema_phase1

if (db $ schema == schema_phase1){
  
  db$connect()
  
  print('Schema is CORRECT! Loading tables ...')
  
  cohort_details_phase1 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "cohort_details")) %>% collect()
  
  dataset_details_phase1 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "dataset_details")) %>% collect()
  
  patient_cohort_df_phase1 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "patient_cohort")) %>% collect()
  
  patient_demographic_df_phase1 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "patient_demographic")) %>% collect()
  
  print('Loaded basic demographics tables!')
  print('Now loading diagnosis, procedure, medication_drug and lab_result... This may take up to ~90 mins')
  
  diagnosis_df_phase1 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "diagnosis")) %>% collect()
  
  procedure_df_phase1 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "procedure")) %>% collect()
  
  medication_drug_df_phase1 <- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "drug")) %>% collect()
  
  lab_result_df_phase1<- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "lab_result")) %>% collect()
  
  encounter_df_phase1<- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "encounter")) %>% collect()
  
  tumor_df_phase1<- db$con %>%
    tbl(dbplyr::in_schema(db$schema, "tumor")) %>% collect()
  
  print('All needed tables are loaded!')
  
  db$disconnect()
  
}else{
  print("Schema is not correct, double check!")
}


# 2. Load all needed codes ####

load_codes_RSVburden()

#3. Select RSV patients and RSV records ####

lab_result_RSV_df_phase1 = lab_result_df_phase1 %>% 
  filter(code %in% rsv_codes) %>%
  filter(!is.na(lab_result_text_val), lab_result_text_val != "Unknown") %>%
  mutate(lab_result_text_val = factor(lab_result_text_val)) %>%
  mutate(code_system = factor(code_system))

summary(lab_result_RSV_df_phase1)

# RSV positive patients
RSV_positive_patients_phase1 = unique(lab_result_RSV_df_phase1 $ patient_id [lab_result_RSV_df_phase1 $ lab_result_text_val == "Positive"])
RSV_negative_patients_phase1 = unique(lab_result_RSV_df_phase1 $ patient_id [lab_result_RSV_df_phase1 $ lab_result_text_val == "Negative"])
all_patients = unique(patient_demographic_df_phase1 $ patient_id)


# 4. Tables deduplication ####

target_df = c('diagnosis','procedure', 'lab_result', 'medication_drug','encounter')

dedup_dfs <- lapply(target_df, function(t) {
  get(paste0(t, '_df_phase1')) %>% distinct()
})

names(dedup_dfs) <- paste0('dedup_', (target_df), '_df_phase1')
list2env(dedup_dfs, envir = .GlobalEnv)


# 5. Load final predictors data, if needed ####

rsv_predictors_df_phase1 = load_phase1_predictors_df()
summary(rsv_predictors_df_phase1)

# Split train and test as it should be split
set.seed(123) 
df = rsv_predictors_df_phase1

trainIndex <- caret::createDataPartition(df $ RSV_test_result, p = .8, list = FALSE, times = 1)
trainData_phase1  <- df[ trainIndex,]
testData_phase1  <- df[-trainIndex,]

summary(testData_phase1) # number of Positive results should be 530 !!!




