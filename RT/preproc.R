#!/usr/bin/env Rscript

library(Rmisc)
library(tidyverse)
library(stringr)
library(scales)
library(grid)
library(ggpubr)
library(MASS)
library(lmerTest)
library(lme4)
library(stats)
library(modelr)
library(plotrix)
library(mgcv)
library(hexbin)
library(formattable)
# library(readr)

rm(list=ls())

clargs <- commandArgs(trailingOnly=TRUE)

metrics_file <- clargs[1]
candidates_file <- clargs[2]
output_file <- clargs[3]
corpus_type <- clargs[4]

# candidates <- c('item','zone','word','length','surprisal', 'surprisal.1',
#                 'surprisal.2',
#                 'position',
#                 'expected_distance', 'kl_divergence', 'head_distance',
#                 'demberg',
#                 'first_dependent_distance', 'first_dependent_correct',
#                 'first_dependent_distance_weight',
#                 'left_dependents_distance_sum', 'left_dependents_count',
#                 'deprel','pos', 'first_dependent_deprel', 'frequency')


big.dundee <- readr::read_csv(candidates_file)

dundee.meta <- read.csv(print(metrics_file)) # %>%
#  dplyr::select(candidates)

scaling_var <- function(data){
  # The input data is a vector
  data <- data}
#as.numeric(data)
#  (data - mean(data,na.rm=TRUE))/sd(data,na.rm=TRUE)
#}

if (corpus_type == "ET") {
  big.dundee2 <- big.dundee %>%
    group_by(item, zone, WorkerId) %>%
    summarise(FFD = sum(FFD),
              GPT = sum(GPT),
              RBT = sum(RBT),
              GD = sum(GD),
              word = word) %>%
    ungroup() %>%
    distinct()
  interest <- c('item','zone','WorkerId', 'FFD', 'GPT', 'RBT', 'GD')
} else {
  big.dundee2 <- big.dundee %>%
    group_by(item, zone, WorkerId) %>%
    summarise(RT = sum(RT),
              word = word) %>%
    ungroup() %>%
    distinct()
  interest <- c('item','zone','WorkerId', 'RT')
}

# big.dundee does not have RT for WNUM=1
dundee <- big.dundee2 %>%
  dplyr::select(interest) %>%
  inner_join(dundee.meta)
  # mutate(surprisal = as.numeric(surprisal),
  #        surprisal.1 = as.numeric(surprisal.1),
  #        surprisal.2 = as.numeric(surprisal.2),
  #        frequency = as.numeric(frequency),
  #        position = as.numeric(position),
  #        head_distance = as.numeric(head_distance),
  #        demberg = as.numeric(demberg),
  #        first_dependent_distance = as.numeric(first_dependent_distance),
  #        first_dependent_distance_weight = as.numeric(first_dependent_distance_weight),
  #        left_dependents_distance_sum = as.numeric(left_dependents_distance_sum),
  #        left_dependents_count = as.numeric(left_dependents_count),
  #        length = as.numeric(length),
  #        WorkerId=as.factor(WorkerId))

#dundee <- dundee[
#  order(dundee[,3], dundee[,1], dundee[,2] ),
#]

# dundee <- dundee %>%
#   mutate(surprisal.s=scaling_var(dundee$surprisal),
#          surprisal.s.1=scaling_var(dundee$surprisal.1),
#          surprisal.s.2=scaling_var(dundee$surprisal.2),
#          frequency.s=scaling_var(dundee$frequency),
#          length.s=scaling_var(dundee$length)) %>%
#   group_by(WorkerId, item) %>%
#   ungroup()

# dundee <- dundee %>%
#   filter(surprisal>-20)

# Get token number
dundee %>%
  dplyr::select(c('item', 'zone', 'word')) %>%
  distinct() %>%
  nrow()  # 24679 tokens


write.csv(dundee, output_file, row.names=FALSE)



