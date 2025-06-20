###################################################################################################################
#
### Importation packages  
library(dplyr)
library(tidyr)

###################################################################################################################
#
### Load Dataset 

run1 <- read.table("sub-001_task-MGT_run-01_events.tsv", sep = "\t", header = TRUE)
str(run1)
run2 <- read.table("sub-001_task-MGT_run-02_events.tsv", sep = "\t", header = TRUE)
run3 <- read.table("sub-001_task-MGT_run-03_events.tsv", sep = "\t", header = TRUE)
run4 <- read.table("sub-001_task-MGT_run-04_events.tsv", sep = "\t", header = TRUE)


txt_gain <- function(run, name) {
  dataset_gain <- run %>%
    select(onset, duration, gain) 
  write.table(dataset_gain , file=paste(name, "txt" , sep = "."), sep=" " , row.names = FALSE , quote = TRUE , col.names = FALSE)
}

run1_gain <- txt_gain(run1, "sub-001_run1_gain")
run1_gain
run2_gain <- txt_gain(run2, "sub-001_run2_gain")
run2_gain
run3_gain <- txt_gain(run3, "sub-001_run3_gain")
run3_gain
run4_gain <- txt_gain(run4, "sub-001_run4_gain")
run4_gain

txt_loss <- function(run, name) {
  dataset_loss <- run %>%
    select(onset, duration, loss)
  write.table(dataset_loss , file=paste(name, "txt" , sep = "."), sep=" " , row.names = FALSE , quote = TRUE , col.names = FALSE)
}

run1_loss <- txt_loss(run1, "sub-001_run1_loss")
run1_loss
run2_loss <- txt_loss(run2, "sub-001_run2_loss")
run2_loss
run3_loss <- txt_loss(run3, "sub-001_run3_loss")
run3_loss
run4_loss <- txt_loss(run4, "sub-001_run4_loss")
run4_loss

txt_intercept <- function(run, name) {
  dataset_intercept <- run %>%
    select(onset, duration) %>%
    mutate(amplitude = 1)
  write.table(dataset_intercept , file=paste(name, "txt" , sep = "."), sep=" " , row.names = FALSE , quote = TRUE , col.names = FALSE)
}

run1_intercept <- txt_intercept(run1, "sub-001_run1_intercept")
run1_intercept
run2_intercept <- txt_intercept(run2, "sub-001_run2_intercept")
run2_intercept
run3_intercept <- txt_intercept(run3, "sub-001_run3_intercept")
run3_intercept
run4_intercept <- txt_intercept(run4, "sub-001_run4_intercept")
run4_intercept

txt_noise <- function(run, name) {
  dataset_noise <- run %>%
    select(onset, duration, RT)
  write.table(dataset_noise , file=paste(name, "txt" , sep = "."), sep=" " , row.names = FALSE , quote = TRUE , col.names = FALSE)
}

run1_noise <- txt_noise(run1, "sub-001_run1_noise")
run1_noise
run2_noise <- txt_noise(run2, "sub-001_run2_noise")
run2_noise
run3_noise <- txt_noise(run3, "sub-001_run3_noise")
run3_noise
run4_noise <- txt_noise(run4, "sub-001_run4_noise")
run4_noise