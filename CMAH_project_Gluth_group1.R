###########################
### CMAH - Student groups #
###########################
### Full code #
####################################
### sebastian.gluth@uni-hamburg.de #
####################################

## Background information ##
##
## the dataset ("data_allTrialTypes") is based on Mechera-Ostrovsky & Gluth, 2018, Scientific Reports
## it can be found on the OSF websites of Kraemer et al., 2021, Psychonomic Bulletin & Review: https://osf.io/y496r/
##
## in this study, participants worked on the so-called remember-and-decide task:
## in every trial, there are 6 different locations on the screen
## in the ENCODING phase, people learn to associate the locations with different food snacks (e.g., chips, chocolate, ...)
## in the DECISION phase, in each trial two locations are highlighted but the snacks themselves are not shown
## so the task is to i) recall the snacks hidden behind the two highlighted locations, and ii) to choose the preferred snack
## in the RECALL phase, in each trial one location is highlighted and participants are asked to recall the snack
## this procedure is repeated 24 times (so-called "blocks"), always with new snack-location associations
##
## in addition to this task, we asked participants to rate each snack in terms of how much they like the snack on a continuous scale
## thus for each snack, we also know the "subjective value" of that snack for each participant
## the task was first introduced in Gluth et al., 2015, Neuron

## Goals of this exercise ##
## 
## 1. get a feeling for the data by analyzing the accuracy as a function of whether both, only one, or none of the snacks was remembered
## 2. fit a logit model (McFadden, 2002, American Economic Review) to the decisions, taking the snack ratings into account
##  --> however, instead of using an in-build function for General Linear Models, use your own softmax choice rule
##  --> we will include an intercept parameter to check whether participants had a left/right choice bias
## 3. fit a second logit model to the decisions that also takes into account whether people have remembered the options
## 4. compare the fits of the models set up in 2. and 3. (both quantitatively and qualitatively)
## 5. fit a sequential sampling model to the decisions and response times; either DDM or LBA

## Some info about the columns of "data_allTrialTypes"
##
## subjID: subject ID
## snack1Val: value of the 1st snack, which is the snack shown on the left
## snack2Val: value of the 2nd snack, which is the snack shown on the right
## recallType: 0 = none of the 2 snacks remembered, 1 = only one snack remembered, 2 = both snacks remembered
## RT: response time in s
## valDiff: value difference between the snacks (in favor of right, so snack2Val - snack1Val)
## choiceRight: whether right option was chosen (= 1) or left option (= 0)


# import packages
library(tidyverse)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# import the data
dataset <- read.delim('data_allTrialTypes', header = TRUE, sep = ",")

## accuracy choosing the higher-valued option
dataset$correct <- 0

dataset$correct <- ifelse((dataset$valDiff > 0 & dataset$choiceRight == 1) |
                            (dataset$valDiff < 0 & dataset$choiceRight == 0), 1, 0)

dataset$correct <- ifelse(dataset$valDiff == 0, NA, dataset$correct)

# mean with group by recallType
data_aggregated <- dataset %>% 
  group_by(subjID, recallType) %>%
  summarise(accuracy = mean(correct, na.rm = T))

# store sample size
N <- length(unique(dataset$subjID))

# add the mean and SD of accuracy for the respective conditions to the dataframe
data_aggregated <- merge(data_aggregated, 
                         data_aggregated %>%
                           group_by(recallType) %>%
                           summarize(mean_accuracy = mean(accuracy),
                                     sd_accuracy = sd(accuracy)),
                         by = "recallType")


data_aggregated %>%
  mutate(recallType = factor(recallType)) %>%
  ggplot(aes(x = recallType, y = accuracy, color = recallType, fill = recallType)) +
  # display the mean accuracy in each condition with a bar
  stat_summary(fun = mean, geom = "bar", width = .5, alpha = .3) +
  # also display individual datapoints
  geom_jitter(width = .3, alpha = .8) +
  # make errorbars with confidence intervals
  geom_errorbar(aes(ymin = mean_accuracy - 1.98 * sd_accuracy,
                    ymax = mean_accuracy + 1.98 * sd_accuracy),
                width = .15, position = position_dodge(.9)) + 
  # hide the legend
  guides(color = F, fill = F) +
  labs(x = "Number of options recalled", y = "Accuracy/Consistency of Choice") +
  theme_minimal() 

