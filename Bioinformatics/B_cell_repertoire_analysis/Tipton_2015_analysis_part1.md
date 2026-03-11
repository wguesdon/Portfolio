Tipton\_2015\_analysis\_part1\_v01-01
================
Compiled: February 29, 2020

``` r
################
# Load libraries
################

library(tidyverse)
library(alakazam)
library(shazam)
library(cowplot)
library(rstatix)
library(cowplot)
library(ggpubr)
```

Abstract
========

Introduction
============

    * Obtained data set from the Antibody Database
    * Firt goal comapring physicco chemical properties of Lupus Ig compared to Health Control sequences

Docker Changeo Analysis
=======================

1.  Connect to Immcantation Docker in the Mac terminal
2.  Parse IMGT output
3.  Remove non-functional sequences
4.  Add Isotype and Cell Subset column
5.  Run clonal analysis

``` bash
# Run Docker in the Mac terminal

cd /Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/AntibodyMap

#####################################
# 1 Connect to docker in the terminal
#####################################

docker run -it -v /Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/AntibodyMap:/data:z kleinstein/immcantation:2.7.0 bash
cd data/
ls
cd HC
ls
>
Tipton_2015_FLU_Bulk_FLU_subject_1_iglblastn_Bulk.fasta
Tipton_2015_FLU_Bulk_FLU_subject_1_iglblastn_Bulk.fasta.gz
Tipton_2015_FLU_Bulk_FLU_subject_1_iglblastn_Bulk.json.gz
Tipton_2015_FLU_Bulk_FLU_subject_1_iglblastn_Bulk.txz
Tipton_2015_FLU_Bulk_FLU_subject_2_iglblastn_Bulk.fasta
Tipton_2015_FLU_Bulk_FLU_subject_2_iglblastn_Bulk.fasta.gz
Tipton_2015_FLU_Bulk_FLU_subject_2_iglblastn_Bulk.json.gz
Tipton_2015_FLU_Bulk_FLU_subject_2_iglblastn_Bulk.txz
Tipton_2015_FLU_Bulk_FLU_subject_3_iglblastn_Bulk.fasta
Tipton_2015_FLU_Bulk_FLU_subject_3_iglblastn_Bulk.fasta.gz
Tipton_2015_FLU_Bulk_FLU_subject_3_iglblastn_Bulk.json.gz
Tipton_2015_FLU_Bulk_FLU_subject_3_iglblastn_Bulk.txz
Tipton_2015_FLU_Bulk_FLU_subject_4_iglblastn_Bulk.fasta
Tipton_2015_FLU_Bulk_FLU_subject_4_iglblastn_Bulk.fasta.gz
Tipton_2015_FLU_Bulk_FLU_subject_4_iglblastn_Bulk.json.gz
Tipton_2015_FLU_Bulk_FLU_subject_4_iglblastn_Bulk.txz
Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.fasta
Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.fasta.gz
Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.json.gz
Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.txz
Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.fasta
Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.fasta.gz
Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.json.gz
Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.txz
Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.fasta
Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.fasta.gz
Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.json.gz
Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.txz
Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.fasta
Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.fasta.gz
Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.json.gz
Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.txz

cd ..
cd cd SLE/
ls
>
Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.txz
Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.txz
Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.txz
Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.txz
Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.txz
Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.txz
Tipton_2015_SLE_Bulk_SLE_subject_7_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_7_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_7_iglblastn_Bulk_fasta.txz
Tipton_2015_SLE_Bulk_SLE_subject_7_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_8_iglblastn_Bulk.fasta
Tipton_2015_SLE_Bulk_SLE_subject_8_iglblastn_Bulk.fasta.gz
Tipton_2015_SLE_Bulk_SLE_subject_8_iglblastn_Bulk.json.gz
Tipton_2015_SLE_Bulk_SLE_subject_8_iglblastn_Bulk.txz

######################
# 2 Parse IMGT output
######################
cd data/HC/
MakeDb.py imgt -i Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.txz -s Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.fasta --regions --scores
>
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:43:29 |Done                | 0.0 min

PROGRESS> 18:43:33 |####################| 100% (18,576) 0.1 min

OUTPUT> Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk_db-pass.tab
  PASS> 1039
  FAIL> 17537
   END> MakeDb

MakeDb.py imgt -i Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.txz -s Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.fasta --regions --scores
>
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:44:17 |Done                | 0.0 min

PROGRESS> 18:44:20 |####################| 100% (10,196) 0.0 min

OUTPUT> Tipton_2015_TET_Bulk_TET_subject_2_iglblastn_Bulk_db-pass.tab
  PASS> 668
  FAIL> 9528
   END> MakeDb

MakeDb.py imgt -i Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.txz -s Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.fasta --regions --scores
>
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:44:59 |Done                | 0.0 min

PROGRESS> 18:45:00 |####################| 100% (3,532) 0.0 min

OUTPUT> Tipton_2015_TET_Bulk_TET_subject_3_iglblastn_Bulk_db-pass.tab
  PASS> 1585
  FAIL> 1947
   END> MakeDb

MakeDb.py imgt -i Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.txz -s Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.fasta --regions --scores
>
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:45:53 |Done                | 0.0 min

PROGRESS> 18:45:59 |####################| 100% (24,445) 0.1 min

OUTPUT> Tipton_2015_TET_Bulk_TET_subject_4_iglblastn_Bulk_db-pass.tab
  PASS> 1810
  FAIL> 22635
   END> MakeDb

cd ..
cd  cd SLE/

MakeDb.py imgt -i Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.txz -s Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.fasta --regions --scores
>
      START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:48:01 |Done                | 0.0 min

PROGRESS> 18:48:06 |####################| 100% (21,893) 0.1 min

OUTPUT> Tipton_2015_SLE_Bulk_SLE_subject_1_iglblastn_Bulk_db-pass.tab
  PASS> 912
  FAIL> 20981
   END> MakeDb

MakeDb.py imgt -i Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.txz -s Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.fasta --regions --scores
>
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:50:56 |Done                | 0.0 min

PROGRESS> 18:51:00 |####################| 100% (15,164) 0.1 min

OUTPUT> Tipton_2015_SLE_Bulk_SLE_subject_2_iglblastn_Bulk_db-pass.tab
  PASS> 1012
  FAIL> 14152
   END> MakeDb
   
MakeDb.py imgt -i Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.txz -s Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.fasta --regions --scores
>  
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:51:39 |Done                | 0.2 min

PROGRESS> 18:52:19 |####################| 100% (172,694) 0.7 min

OUTPUT> Tipton_2015_SLE_Bulk_SLE_subject_3_iglblastn_Bulk_db-pass.tab
  PASS> 8162
  FAIL> 164532
   END> MakeDb

MakeDb.py imgt -i Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.txz -s Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.fasta --regions --scores
>
      START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:53:04 |Done                | 0.0 min

PROGRESS> 18:53:14 |####################| 100% (39,846) 0.2 min

OUTPUT> Tipton_2015_SLE_Bulk_SLE_subject_4_iglblastn_Bulk_db-pass.tab
  PASS> 959
  FAIL> 38887
   END> MakeDb

MakeDb.py imgt -i Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.txz -s Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.fasta --regions --scores
>
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:53:42 |Done                | 0.0 min

PROGRESS> 18:53:48 |####################| 100% (25,157) 0.1 min

OUTPUT> Tipton_2015_SLE_Bulk_SLE_subject_5_iglblastn_Bulk_db-pass.tab
  PASS> 1919
  FAIL> 23238
   END> MakeDb


MakeDb.py imgt -i Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.txz -s Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.fasta --regions --scores
>
       START> MakeDb
     ALIGNER> IMGT
ALIGNER_FILE> Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.txz
    SEQ_FILE> Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk.fasta
     ASIS_ID> False
     PARTIAL> False
      SCORES> True
     REGIONS> True
    JUNCTION> False

PROGRESS> 18:54:24 |Done                | 0.0 min

PROGRESS> 18:54:26 |####################| 100% (8,558) 0.0 min

OUTPUT> Tipton_2015_SLE_Bulk_SLE_subject_6_iglblastn_Bulk_db-pass.tab
  PASS> 2997
  FAIL> 5561
   END> MakeDb
```

**A majority of the sequecnes failed. This could be caused by a difference in the pipeline check the in the paper method what sequencing methd was used.**

Library preparation: NGS data is deposited at the NCBI sequence read archive (SRA) study accession, SRP057017. Total cellular RNA was isolated from the number of cells outlined in Table 1 using the RNeasy Micro kit by following the manufacturer's protocol (Qiagen). Approximately 2 ng of RNA was subjected to reverse transcription using the iScript cDNA synthesis kit (BioRad). Aliquots of the resulting single-stranded cDNA products were mixed with 50 nM of VH1-VH7 FR1 specific primers and 250 nM Cα, Cμ, and Cγ specific primers preceded by the respective Illumina nextera sequencing tag (oligonucleotide sequences listed below) in a 25 μl PCR reaction (using 4 αl template cDNA) using Invitrogen's High Fidelity Platinum PCR Supermix (Invitrogen).

Lod each file, edit table and fuse
==================================

-   Load the files
-   Add the variables
    -   Donor
    -   Health Status HC or SLE
    -   Isotype
-   Fuse all the table in one master table
-   Run the rest of Immcantation pipeline on the master table

``` r
HC1  <- readChangeoDb("/Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/AntibodyMap/HC/Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk_db-pass.tab")
```

``` r
directory <- "/Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/AntibodyMap/HC/"
files <- dir(directory, pattern = "*.tab")
```

``` r
desired_length <- 4
HC_list <- vector(mode = "list", length = desired_length)
```

``` r
# #https://stackoverflow.com/questions/33177118/append-a-data-frame-to-a-list
# #https://stackoverflow.com/questions/17499013/how-do-i-make-a-list-of-data-frames
# counter <- 1
# for (i in files){ 
#     file_path <- paste0(directory,i)
#     #print(file_path) 
#     HC_list[counter] <- list(readChangeoDb(file_path))
#     counter <- counter + 1
# }
```

``` r
#https://stackoverflow.com/questions/33177118/append-a-data-frame-to-a-list
#https://stackoverflow.com/questions/17499013/how-do-i-make-a-list-of-data-frames
counter <- 1
for (i in files){ 
    file_path <- paste0(directory,i)
    #print(file_path) 
    HC_list[counter] <- list(readChangeoDb(file_path))
    HC_list[[counter]] <-  HC_list[[counter]] %>%
        dplyr::mutate(
            Donor = counter,
            Status = "HC"
        )
    counter <- counter + 1
}
```

``` r
#glimpse(empty_list)
```

``` r
#db <- empty_list[[1]]
#View(db)
```

``` r
# #dir(directory, pattern = "*.tab")
# file_path <- paste0(directory,"Tipton_2015_TET_Bulk_TET_subject_1_iglblastn_Bulk_db-pass.tab")
# db <- readChangeoDb(file_path)
# glimpse(db)
```

``` r
HC <- HC_list[[1]]
for (i in c(2:length(HC_list))){
    #print(i)
    HC <- bind_rows(HC, HC_list[[i]])
}
```

``` r
#glimpse(HC_list[[4]])
```

``` r
#glimpse(HC)
directory <- "/Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/Immcantation/"
saveRDS(HC, "Output/R_object/all_HC.RDS")
writeChangeoDb(HC, paste0(directory,"all_HC.tab"))
```

``` r
directory <- "/Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/AntibodyMap/SLE/"
files <- dir(directory, pattern = "*.tab")
#files
```

``` r
desired_length <- length(files)
SLE_list <- vector(mode = "list", length = desired_length)
```

``` r
counter <- 1
for (i in files){ 
    file_path <- paste0(directory,i)
    #print(file_path) 
    SLE_list[counter] <- list(readChangeoDb(file_path))
    SLE_list[[counter]] <-  SLE_list[[counter]] %>%
        dplyr::mutate(
            Donor = counter,
            Status = "SLE"
        )
    counter <- counter + 1
}
```

``` r
SLE <- SLE_list[[1]]
for (i in c(2:length(SLE_list))){
    #print(i)
    SLE <- bind_rows(SLE, SLE_list[[i]])
}
```

``` r
#glimpse(SLE)
directory <- "/Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/Immcantation/"
saveRDS(SLE, "Output/R_object/all_SLE.RDS")
writeChangeoDb(SLE, paste0(directory,"all_SLE.tab"))
```

``` r
SLE <- SLE %>% filter(FUNCTIONAL == "TRUE")
HC <- HC %>% filter(FUNCTIONAL == "TRUE")

all_samples <- bind_rows(HC, SLE)
directory <- "/Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/Immcantation/"
saveRDS(all_samples, "Output/R_object/all_samples_pre_threshold.RDS")
writeChangeoDb(all_samples, paste0(directory,"all_samples.tab"))
```

Run the clonal analysis on combined dataset
===========================================

``` bash
#######################
# 4 Determine threshold
#######################

docker run -it -v /Users/william/Documents/GitHub/GitHub_Data/Tipton_2015_SLE_vs_HC/Immcantation:/data:z kleinstein/immcantation:2.7.0 bash
cd data/
ls
>
all_HC.tab  all_samples.tab  all_SLE.tab

shazam-threshold -d all_HC.tab
>
THRESHOLD_AVG> 0.0674195

shazam-threshold -d all_SLE.tab
>
THRESHOLD_AVG> 0.0674195

shazam-threshold -d all_samples.tab 
>
THRESHOLD_AVG> 0.0592632
```

Why is the threshold EXACTLY the same for SLE and HC?
=====================================================

Check that the 2 df are different

``` r
glimpse(HC)
```

    ## Observations: 33,098
    ## Variables: 44
    ## $ SEQUENCE_ID        <chr> "Query_712376", "Query_884469", "Query_470102", "Q…
    ## $ SEQUENCE_INPUT     <chr> "CGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGT…
    ## $ FUNCTIONAL         <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
    ## $ IN_FRAME           <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
    ## $ STOP               <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, F…
    ## $ MUTATED_INVARIANT  <chr> "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", …
    ## $ INDELS             <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FA…
    ## $ V_CALL             <chr> "Homsap IGHV1-18*01 F", "Homsap IGHV3-21*01 F,Homs…
    ## $ D_CALL             <chr> "Homsap IGHD7-27*01 F", "Homsap IGHD4-17*01 F", "H…
    ## $ J_CALL             <chr> "Homsap IGHJ6*02 F", "Homsap IGHJ6*02 F", "Homsap …
    ## $ SEQUENCE_VDJ       <chr> "CGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGT…
    ## $ SEQUENCE_IMGT      <chr> "....CGCAGCTGGTGCAGTCTGGGGCT...GAGGTGAAGAAGCCTGGGG…
    ## $ V_SEQ_START        <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ V_SEQ_LENGTH       <int> 292, 250, 281, 240, 244, 249, 256, 266, 251, 289, …
    ## $ V_GERM_START_VDJ   <int> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_GERM_LENGTH_VDJ  <int> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_GERM_START_IMGT  <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ V_GERM_LENGTH_IMGT <int> 320, 320, 320, 315, 314, 317, 319, 318, 318, 318, …
    ## $ NP1_LENGTH         <int> 0, 15, 10, 0, 2, 3, 3, 2, 5, 6, 4, 14, 11, 2, 7, 4…
    ## $ D_SEQ_START        <int> 293, 266, 292, 241, 247, 253, 260, 269, 257, 296, …
    ## $ D_SEQ_LENGTH       <int> 6, 16, 14, 10, 16, 7, 13, 12, 13, 12, 7, 13, 23, 5…
    ## $ D_GERM_START       <int> 6, 1, 11, 14, 6, 5, 4, 8, 4, 6, 22, 11, 6, 18, 11,…
    ## $ D_GERM_LENGTH      <int> 6, 16, 14, 10, 16, 7, 13, 12, 13, 12, 7, 13, 23, 5…
    ## $ NP2_LENGTH         <int> 2, 11, 7, 6, 7, 8, 2, 5, 5, 1, 7, 1, 2, 6, 16, 2, …
    ## $ J_SEQ_START        <int> 301, 293, 313, 257, 270, 268, 275, 286, 275, 309, …
    ## $ J_SEQ_LENGTH       <int> 50, 48, 29, 40, 44, 43, 44, 38, 46, 35, 8, 52, 47,…
    ## $ J_GERM_START       <int> 4, 5, 11, 13, 9, 8, 7, 13, 8, 6, 11, 2, 5, 2, 14, …
    ## $ J_GERM_LENGTH      <int> 50, 48, 29, 40, 44, 43, 44, 38, 46, 35, 8, 52, 47,…
    ## $ JUNCTION           <chr> "TGTGCGAGAGATGGGGACAACTACTACTACTACGGTATGGACGTCTGG"…
    ## $ JUNCTION_LENGTH    <int> 48, 81, 51, 42, 54, 51, 54, 48, 57, 42, 30, 66, 72…
    ## $ GERMLINE_IMGT      <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_SCORE            <dbl> 1388, 1160, 1162, 892, 1097, 1206, 1159, 1106, 117…
    ## $ V_IDENTITY         <dbl> 0.9894, 0.9793, 0.9194, 0.8650, 0.9504, 0.9959, 0.…
    ## $ J_SCORE            <dbl> 184, 197, 78, 143, 125, 151, 142, 106, 175, 83, 27…
    ## $ J_IDENTITY         <dbl> 0.8302, 0.8654, 0.6667, 0.7500, 0.7115, 0.7800, 0.…
    ## $ FWR1_IMGT          <chr> "....CGCAGCTGGTGCAGTCTGGGGCT...GAGGTGAAGAAGCCTGGGG…
    ## $ FWR2_IMGT          <chr> "ATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGAT…
    ## $ FWR3_IMGT          <chr> "AACTATGCACAGAAGCTCCAG...GGCAGAGTCACCATGACCACAGACA…
    ## $ FWR4_IMGT          <chr> "TGGGACCAAGGGCCCATCGGCGCA", "TGGGGCCAAGGGCCCATCGGC…
    ## $ CDR1_IMGT          <chr> "GGTTACACCTTT............ACCAGCTATGGT", "GGATTCACC…
    ## $ CDR2_IMGT          <chr> "ATCAGCGCTTAC......AATGGTAACACA", "ATTAGTAGTAGT...…
    ## $ CDR3_IMGT          <chr> "GCGAGAGATGGGGACAACTACTACTACTACGGTATGGACGTC", "GCG…
    ## $ Donor              <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ Status             <chr> "HC", "HC", "HC", "HC", "HC", "HC", "HC", "HC", "H…

``` r
#View(HC)
```

``` r
glimpse(SLE)
```

    ## Observations: 57,084
    ## Variables: 44
    ## $ SEQUENCE_ID        <chr> "Query_712376", "Query_884469", "Query_470102", "Q…
    ## $ SEQUENCE_INPUT     <chr> "CGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGT…
    ## $ FUNCTIONAL         <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
    ## $ IN_FRAME           <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
    ## $ STOP               <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, F…
    ## $ MUTATED_INVARIANT  <chr> "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", …
    ## $ INDELS             <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FA…
    ## $ V_CALL             <chr> "Homsap IGHV1-18*01 F", "Homsap IGHV3-21*01 F,Homs…
    ## $ D_CALL             <chr> "Homsap IGHD7-27*01 F", "Homsap IGHD4-17*01 F", "H…
    ## $ J_CALL             <chr> "Homsap IGHJ6*02 F", "Homsap IGHJ6*02 F", "Homsap …
    ## $ SEQUENCE_VDJ       <chr> "CGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGT…
    ## $ SEQUENCE_IMGT      <chr> "....CGCAGCTGGTGCAGTCTGGGGCT...GAGGTGAAGAAGCCTGGGG…
    ## $ V_SEQ_START        <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ V_SEQ_LENGTH       <int> 292, 250, 281, 240, 244, 249, 256, 266, 251, 289, …
    ## $ V_GERM_START_VDJ   <int> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_GERM_LENGTH_VDJ  <int> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_GERM_START_IMGT  <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ V_GERM_LENGTH_IMGT <int> 320, 320, 320, 315, 314, 317, 319, 318, 318, 318, …
    ## $ NP1_LENGTH         <int> 0, 15, 10, 0, 2, 3, 3, 2, 5, 6, 4, 14, 11, 2, 7, 4…
    ## $ D_SEQ_START        <int> 293, 266, 292, 241, 247, 253, 260, 269, 257, 296, …
    ## $ D_SEQ_LENGTH       <int> 6, 16, 14, 10, 16, 7, 13, 12, 13, 12, 7, 13, 23, 5…
    ## $ D_GERM_START       <int> 6, 1, 11, 14, 6, 5, 4, 8, 4, 6, 22, 11, 6, 18, 11,…
    ## $ D_GERM_LENGTH      <int> 6, 16, 14, 10, 16, 7, 13, 12, 13, 12, 7, 13, 23, 5…
    ## $ NP2_LENGTH         <int> 2, 11, 7, 6, 7, 8, 2, 5, 5, 1, 7, 1, 2, 6, 16, 2, …
    ## $ J_SEQ_START        <int> 301, 293, 313, 257, 270, 268, 275, 286, 275, 309, …
    ## $ J_SEQ_LENGTH       <int> 50, 48, 29, 40, 44, 43, 44, 38, 46, 35, 8, 52, 47,…
    ## $ J_GERM_START       <int> 4, 5, 11, 13, 9, 8, 7, 13, 8, 6, 11, 2, 5, 2, 14, …
    ## $ J_GERM_LENGTH      <int> 50, 48, 29, 40, 44, 43, 44, 38, 46, 35, 8, 52, 47,…
    ## $ JUNCTION           <chr> "TGTGCGAGAGATGGGGACAACTACTACTACTACGGTATGGACGTCTGG"…
    ## $ JUNCTION_LENGTH    <int> 48, 81, 51, 42, 54, 51, 54, 48, 57, 42, 30, 66, 72…
    ## $ GERMLINE_IMGT      <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_SCORE            <dbl> 1388, 1160, 1162, 892, 1097, 1206, 1159, 1106, 117…
    ## $ V_IDENTITY         <dbl> 0.9894, 0.9793, 0.9194, 0.8650, 0.9504, 0.9959, 0.…
    ## $ J_SCORE            <dbl> 184, 197, 78, 143, 125, 151, 142, 106, 175, 83, 27…
    ## $ J_IDENTITY         <dbl> 0.8302, 0.8654, 0.6667, 0.7500, 0.7115, 0.7800, 0.…
    ## $ FWR1_IMGT          <chr> "....CGCAGCTGGTGCAGTCTGGGGCT...GAGGTGAAGAAGCCTGGGG…
    ## $ FWR2_IMGT          <chr> "ATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGAT…
    ## $ FWR3_IMGT          <chr> "AACTATGCACAGAAGCTCCAG...GGCAGAGTCACCATGACCACAGACA…
    ## $ FWR4_IMGT          <chr> "TGGGACCAAGGGCCCATCGGCGCA", "TGGGGCCAAGGGCCCATCGGC…
    ## $ CDR1_IMGT          <chr> "GGTTACACCTTT............ACCAGCTATGGT", "GGATTCACC…
    ## $ CDR2_IMGT          <chr> "ATCAGCGCTTAC......AATGGTAACACA", "ATTAGTAGTAGT...…
    ## $ CDR3_IMGT          <chr> "GCGAGAGATGGGGACAACTACTACTACTACGGTATGGACGTC", "GCG…
    ## $ Donor              <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ Status             <chr> "SLE", "SLE", "SLE", "SLE", "SLE", "SLE", "SLE", "…

``` r
#View(SLE)
```

``` r
glimpse(all_samples)
```

    ## Observations: 90,182
    ## Variables: 44
    ## $ SEQUENCE_ID        <chr> "Query_712376", "Query_884469", "Query_470102", "Q…
    ## $ SEQUENCE_INPUT     <chr> "CGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGT…
    ## $ FUNCTIONAL         <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
    ## $ IN_FRAME           <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
    ## $ STOP               <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, F…
    ## $ MUTATED_INVARIANT  <chr> "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", …
    ## $ INDELS             <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FA…
    ## $ V_CALL             <chr> "Homsap IGHV1-18*01 F", "Homsap IGHV3-21*01 F,Homs…
    ## $ D_CALL             <chr> "Homsap IGHD7-27*01 F", "Homsap IGHD4-17*01 F", "H…
    ## $ J_CALL             <chr> "Homsap IGHJ6*02 F", "Homsap IGHJ6*02 F", "Homsap …
    ## $ SEQUENCE_VDJ       <chr> "CGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGT…
    ## $ SEQUENCE_IMGT      <chr> "....CGCAGCTGGTGCAGTCTGGGGCT...GAGGTGAAGAAGCCTGGGG…
    ## $ V_SEQ_START        <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ V_SEQ_LENGTH       <int> 292, 250, 281, 240, 244, 249, 256, 266, 251, 289, …
    ## $ V_GERM_START_VDJ   <int> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_GERM_LENGTH_VDJ  <int> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_GERM_START_IMGT  <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ V_GERM_LENGTH_IMGT <int> 320, 320, 320, 315, 314, 317, 319, 318, 318, 318, …
    ## $ NP1_LENGTH         <int> 0, 15, 10, 0, 2, 3, 3, 2, 5, 6, 4, 14, 11, 2, 7, 4…
    ## $ D_SEQ_START        <int> 293, 266, 292, 241, 247, 253, 260, 269, 257, 296, …
    ## $ D_SEQ_LENGTH       <int> 6, 16, 14, 10, 16, 7, 13, 12, 13, 12, 7, 13, 23, 5…
    ## $ D_GERM_START       <int> 6, 1, 11, 14, 6, 5, 4, 8, 4, 6, 22, 11, 6, 18, 11,…
    ## $ D_GERM_LENGTH      <int> 6, 16, 14, 10, 16, 7, 13, 12, 13, 12, 7, 13, 23, 5…
    ## $ NP2_LENGTH         <int> 2, 11, 7, 6, 7, 8, 2, 5, 5, 1, 7, 1, 2, 6, 16, 2, …
    ## $ J_SEQ_START        <int> 301, 293, 313, 257, 270, 268, 275, 286, 275, 309, …
    ## $ J_SEQ_LENGTH       <int> 50, 48, 29, 40, 44, 43, 44, 38, 46, 35, 8, 52, 47,…
    ## $ J_GERM_START       <int> 4, 5, 11, 13, 9, 8, 7, 13, 8, 6, 11, 2, 5, 2, 14, …
    ## $ J_GERM_LENGTH      <int> 50, 48, 29, 40, 44, 43, 44, 38, 46, 35, 8, 52, 47,…
    ## $ JUNCTION           <chr> "TGTGCGAGAGATGGGGACAACTACTACTACTACGGTATGGACGTCTGG"…
    ## $ JUNCTION_LENGTH    <int> 48, 81, 51, 42, 54, 51, 54, 48, 57, 42, 30, 66, 72…
    ## $ GERMLINE_IMGT      <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ V_SCORE            <dbl> 1388, 1160, 1162, 892, 1097, 1206, 1159, 1106, 117…
    ## $ V_IDENTITY         <dbl> 0.9894, 0.9793, 0.9194, 0.8650, 0.9504, 0.9959, 0.…
    ## $ J_SCORE            <dbl> 184, 197, 78, 143, 125, 151, 142, 106, 175, 83, 27…
    ## $ J_IDENTITY         <dbl> 0.8302, 0.8654, 0.6667, 0.7500, 0.7115, 0.7800, 0.…
    ## $ FWR1_IMGT          <chr> "....CGCAGCTGGTGCAGTCTGGGGCT...GAGGTGAAGAAGCCTGGGG…
    ## $ FWR2_IMGT          <chr> "ATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGAT…
    ## $ FWR3_IMGT          <chr> "AACTATGCACAGAAGCTCCAG...GGCAGAGTCACCATGACCACAGACA…
    ## $ FWR4_IMGT          <chr> "TGGGACCAAGGGCCCATCGGCGCA", "TGGGGCCAAGGGCCCATCGGC…
    ## $ CDR1_IMGT          <chr> "GGTTACACCTTT............ACCAGCTATGGT", "GGATTCACC…
    ## $ CDR2_IMGT          <chr> "ATCAGCGCTTAC......AATGGTAACACA", "ATTAGTAGTAGT...…
    ## $ CDR3_IMGT          <chr> "GCGAGAGATGGGGACAACTACTACTACTACGGTATGGACGTC", "GCG…
    ## $ Donor              <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,…
    ## $ Status             <chr> "HC", "HC", "HC", "HC", "HC", "HC", "HC", "HC", "H…

``` r
#View(all_samples)
```

Did not used parse select to remove non functional sequence but instead directly use dplyr. After verification it appears that the dataset are clearly differents

``` bash
####################
# 5 Clonal threshold
####################

changeo-clone -d all_samples.tab -x 0.0592632

>
START
   1: ParseDb select           14:01 01/22/20
   2: DefineClones             14:01 01/22/20
   3: CreateGermlines          14:05 01/22/20
   4: ParseLog                 14:06 01/22/20
   5: Compressing files        14:06 01/22/20
DONE
```
