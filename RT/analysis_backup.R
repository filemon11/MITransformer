library(Rmisc)
library(tidyverse)
library(stringr)
library(scales)
library(grid)
library(ggpubr)
library(MASS)
library(lme4)
library(stats)
library(modelr)
library(plotrix)
library(mgcv)
library(hexbin)
library(formattable)


rm(list=ls())

args = commandArgs(trailingOnly=TRUE)


data <- read.csv(args[1])

###############

# ---------------------- Step 1 ----------------------
en.s0 = lmer(RT ~ logfreq.s + wlen.s + 
               (1|WorkerId) + (1|item),
             data=na.omit(data),
             REML=FALSE)
summary(en.s0)

# ---------------------- Step 2 ----------------------
en.s1 = lmer(RT ~ logfreq.s + wlen.s + logp.s +
               (1|WorkerId) + (1|item),
             data=na.omit(data),
             REML=FALSE)
summary(en.s1)

anova(en.s0, en.s1)  # Significant?

###############

# Compute a pearson like correlation by measuring the correlation
# between the log probs and the rt averaged over the readers.
# Plot the data and check that the relation is approx linear.


means <- aggregate(cbind(RT, logp.s) ~ item + zone,
                   data = data, FUN = mean, na.rm = TRUE)

cor.test(means$RT, means$logp.s, method="pearson")

png(filename=args[2])
plot(means$logp.s, means$RT)+
  abline(lm(means$RT ~ means$logp.s))
dev.off()

