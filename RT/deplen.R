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
library(MuMIn)
library(ggrepel)
library(data.table)
library(comprehenr)
library(ggplot2)
library(dplyr)
library(car)
library(sjPlot)
library(inlcolor)

data_dir <- "./data/EWT_RT_train_preprocessed_exp1_4_0_headcost.csv"
data <- read.csv(data_dir)
# data <- data[!(data["deprel"] == "punct"),]
# data <- data[!(data["pos"] == "$"),]
# data <- data[!(data["pos"] == "#"),]
# data <- data[!(data['pos'] == '"'),]
# data <- data[!(data['pos'] == "''"),]

# data <- data[!(data['pos'] == "DT"),]
# data <- data[!(data['pos'] == "CC"),]
# data <- data[!(data['pos'] == "EX"),]
# data <- data[!(data['pos'] == "IN"),]
# data <- data[!(data['pos'] == "JJ"),]
# data <- data[!(data['pos'] == "MD"),]
# data <- data[!(data['pos'] == "POS"),]
# data <- data[!(data['pos'] == "RB"),]
# data <- data[!(data['pos'] == "RP"),]
# data <- data[!(data['pos'] == "TO"),]
# data <- data[!(data['pos'] == "PRP"),]
# data <- data[!(data['pos'] == "CD"),]
# data <- data[!(data['pos'] == "FW"),]
# data <- data[!(data['pos'] == "LS"),]
# data <- data[!(data['pos'] == "``"),]
# data <- data[!(data['pos'] == ":"),]
# data <- data[!(data['pos'] == "."),]

#data$costs <- pmax(0, data$costs-1)


cor.test(data[["costs"]], data[["attention_distance"]], method="pearson")

# get unique levels in a stable order
pos_levels <- sort(unique(data$pos))

# assign colors based on levels
#cols <- c("brown", "darkgreen")
cols <- get_colors(length(pos_levels))

# map colors to data
clrs <- cols[factor(data$pos, levels = pos_levels)]

#shapes <- c(0,1)
#pchs <- shapes[factor(data$pos, levels = pos_levels)]

svg("dependencies_full.svg")
# plot
plot(data$attention_distance, data$costs, col = clrs, ylim=c(0, 100), xlim=c(0, 40), xlab="Dependency length", ylab="Attention distance")

# add legend
legend("topright",
       legend = pos_levels,
       col = cols,
       pch = 1, bg="white")   # match plot symbol

dev.off()

