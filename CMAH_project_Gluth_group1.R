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


# clear working environment
rm(list=ls())

# clear all plots
if(!is.null(dev.list())) dev.off()

### load required packages
## for settling
library(tidyverse)
library(tidyquant)
## for plotting
library(ggplot2)
library(ggdist)
library(ggstatsplot)
## for modeling
library(dfoptim)
library(rtdists)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# import the data
dataset <- read.delim('data_allTrialTypes', header = TRUE, sep = ",")

#------------------------------------------------------------------------------#
#### Task 1 ####

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
  stat_summary(fun = mean, geom = "bar", width = .3, alpha = .3) +
  # also display individual datapoints
  geom_jitter(width = .2, alpha = .8) +
  # make errorbars with confidence intervals
  geom_errorbar(aes(ymin = mean_accuracy - (1.98 * (sd_accuracy / sqrt(N))),
                    ymax = mean_accuracy + (1.98 * (sd_accuracy / sqrt(N)))),
                width = .15, position = position_dodge(.9)) + 
  # hide the legend
  guides(color = F, fill = F) +
  labs(x = "Number of options recalled", y = "Accuracy/Consistency of Choice") +
  theme_minimal()

#------------------------------------------------------------------------------#
#### Task 2: Fit a logit model to the data ####

#### simulate some data
steps <- seq(-3, 3, 1)

# returns the choice probability of the given value difference (valDiffSim)
p_right <- function(valDiffSim, rightBias = 1, choiceSensitivity = 1) {
  return(1 / (1 + exp(-rightBias - choiceSensitivity * valDiffSim)))
}

probabilities_sim <- p_right(valDiffSim = steps,
                             rightBias = 0,
                             choiceSensitivity = 1)
plot(x = steps, y = probabilities_sim)


#### build a likelihood function

# returns the negative log likelihood for a given value difference and choice
# according to the logit model
choiceModel1 <- function(valDiff, choiceRight, rightBias = 1, choiceSensitivity = 1) {
  
  p_right <- 1 / (1 + exp(-rightBias - choiceSensitivity * valDiff))
  neg_log_lik <- - (sum(log(p_right[choiceRight == 1])) + sum(log(1 - p_right[choiceRight == 0])))
  
  return(neg_log_lik)
}

# test function
choice_right_sim <- sample(c(0, 1), length(steps), replace = T)
log_lik_sim <- choiceModel1(steps, choice_right_sim, choiceSensitivity = 1, rightBias = 1)

# plausible choices for the value diffs in steps (choose left for a negative
# difference, right for 0 and a positive difference)
choice_right_sim_plausible <- c(0, 0, 0, 1, 1, 1, 1)
# reverse: very implausible choices
choice_right_sim_implausible <- c(1, 1, 1, 0, 0, 0, 0)

# compute the log likelihood for the two choices
log_lik_plausible <- choiceModel1(steps, choice_right_sim_plausible, choiceSensitivity = 1, rightBias = 0)
log_lik_implausible <- choiceModel1(steps, choice_right_sim_implausible, choiceSensitivity = 1, rightBias = 0)

#### Task 2.2
#### fit the logit model to the data using grid search

# initialize some values 
bias <- seq(-2, 2, .1)
choiceSensitivity <- seq(-2, 2, .1)
subjects <- unique(dataset$subjID)
best_bias <- rep(NA, length(subjects))
best_sensitivity <- rep(NA, length(subjects))
best_nLL <- rep(1/0, length(subjects))

bestParamsGridSearch <- matrix(data = 1/0, nrow = length(subjects), ncol = 3)


# fit the model separately to all subjects by trying different values for 
# both parameters
for (s_idx in 1:length(subjects)) {
  subj_tmp <- subjects[s_idx]
  
  for (bias_idx in 1:length(bias)) {
    bias_tmp <- bias[bias_idx]
    
    for (choice_sens_idx in 1:length(choiceSensitivity)) {
      choice_sens_tmp <- choiceSensitivity[choice_sens_idx]
      
      # feed the data irrespective of conditions
      choiceRight <- dataset$choiceRight[dataset$subjID == subj_tmp]
      valDiff <- dataset$valDiff[dataset$subjID == subj_tmp]
      
      # compute the negative log likelihood of the parameter values in the current
      # iteration for subj_tmp
      neg_log_lik <- choiceModel1(valDiff = valDiff, choiceRight = choiceRight,
                                  rightBias = bias_tmp, choiceSensitivity = choice_sens_tmp)
      
      # update the best parameter values if the values return a smaller (i.e., better)
      # negative log likelihood
      if (neg_log_lik < bestParamsGridSearch[s_idx, 3]) {
        bestParamsGridSearch[s_idx, 1] <- bias_tmp
        bestParamsGridSearch[s_idx, 2] <- choice_sens_tmp
        bestParamsGridSearch[s_idx, 3] <- neg_log_lik
      }
    }
  }
}




#### Task 2.3
#### Fit the model using an optimization algorithm (i.e., Nelder-Mead optimization)
#nmk_results <- vector("list", length = N)
bestParamsnmkSearch <- matrix(data = 1/0, nrow = length(subjects), ncol = 3)

for (s_idx in 1:length(subjects)) {
  subj_tmp <- subjects[s_idx]
  
  # starting value for two parameters
  startVal <- c(bestParamsGridSearch[s_idx, 1], bestParamsGridSearch[s_idx, 2])
  
  # feed data
  choiceRight <- dataset$choiceRight[dataset$subjID == subj_tmp]
  valDiff <- dataset$valDiff[dataset$subjID == subj_tmp]
  
  # use nmk function of R-package dfoptim
  fitModel <- nmk(par = startVal, fn = function(par) {
    choiceModel1(rightBias = par[1], choiceSensitivity = par[2],
                 choiceRight = choiceRight, valDiff = valDiff)
    })
  
  # save outputs and print convergence messages
  bestParamsnmkSearch[s_idx, 1] <- fitModel$par[1]
  bestParamsnmkSearch[s_idx, 2] <- fitModel$par[2]
  bestParamsnmkSearch[s_idx, 3] <- fitModel$value

  print(fitModel$message)
}



#### Task 2.4
#### Have a look at the parameter estimates

# choiceSensitivity
t.test(bestParamsnmkSearch[,1] - 0)

# rightBiase
t.test(bestParamsnmkSearch[,2] - 0)


# test whether the built-in glm() function leads to the same results for the
# first subjects
bestParamsglmSearch <- matrix(data = 1/0, nrow = length(subjects), ncol = 3)


for (s_idx in 1:length(subjects)) {
  subj_tmp <- subjects[s_idx]
  
  # use glm function
  fitModel <- glm(choiceRight ~ valDiff, family = binomial(link = "logit"),
                  data = dataset[dataset$subjID == subj_tmp, ])
  
  # save outputs and print convergence messages
  bestParamsglmSearch[s_idx, 1] <- fitModel$coefficients[[1]]
  bestParamsglmSearch[s_idx, 2] <- fitModel$coefficients[[2]]
  bestParamsglmSearch[s_idx, 3] <- fitModel$aic

}

# it does, yay :)

plot(x = bestParamsnmkSearch[,2], y = bestParamsglmSearch[,2],
     xlab = "Sensitivity parameter of choice model 1",
     ylab = "Slope coefficient of logistic regression",
     main = "Comparison between model and regression parameters")

data_summarised <- dataset %>% 
  group_by(subjID) %>%
  summarise(frequency_choiceRight = mean(choiceRight, na.rm = T))

plot(x = bestParamsnmkSearch[,1], y = data_summarised$frequency_choiceRight,
     col = "red",
     xlab = "Bias parameter of choice model 1",
     ylab = "Frequency of choosing right option",
     main = "Comparison model-based and descriptive choice bias")


#------------------------------------------------------------------------------#
#### Day 4 ####
#### Task 3
#### Assume that the choiceSensitivity parameter differs across conditions

choiceModel2 <- function(valDiff, choiceRight, recall_condition, 
                         rightBias = 1, choiceSensitivity = c(1, 1, 1)) {
  choiceSensitivity_vector <- choiceSensitivity[1] * (recallType==0)+
    choiceSensitivity[2] * (recallType==1)+
    choiceSensitivity[3] * (recallType==2)
  p_right <- 1 / (1 + exp(-rightBias - choiceSensitivity_vector * valDiff))
  neg_log_lik <- - (sum(log(p_right[choiceRight == 1])) + 
                      sum(log(1 - p_right[choiceRight == 0])))
  return(neg_log_lik)
}


#### Task 3.2
bestParamsnmkSearch_conditions <- matrix(data = 1/0, nrow = length(subjects), ncol = 5)


for (s_idx in 1:length(subjects)) {
  subj_tmp <- subjects[s_idx]
  
  # starting value for two parameters, vector for sensitivity
  startVal <- c(bestParamsGridSearch[s_idx, 1], rep(bestParamsGridSearch[s_idx, 2],3))
  
  # feed data
  choiceRight <- dataset$choiceRight[dataset$subjID == subj_tmp]
  valDiff <- dataset$valDiff[dataset$subjID == subj_tmp]
  recallType <- dataset$recallType[dataset$subjID == subj_tmp]
  
  # use nmk function of R-package dfoptim
  fitModel <- nmk(par = startVal, fn = function(par) 
    choiceModel2(rightBias = par[1], choiceSensitivity = par[2:4],
                 choiceRight = choiceRight, valDiff = valDiff, recall_condition = recallType))
  
  # save outputs and print convergence messages
  for (i in 1:4) {
    bestParamsnmkSearch_conditions[s_idx, i] <- fitModel$par[i]
  }
  bestParamsnmkSearch_conditions[s_idx, i+1] <- fitModel$value
  
  print(fitModel$message)
}

#### Task 3.3
## compare the parameters across conditions with plot
df_data_plot <- data.frame(recallType = c(rep(0,90), rep(1,90), rep(2,90)),
                           choiceSensitivity = c(bestParamsnmkSearch_conditions[,2:4]))

plot(x = df_data_plot$recallType, y = df_data_plot$choiceSensitivity, ylim = c(-0.2,1.0))

ggbetweenstats(df_data_plot, x = recallType, y = choiceSensitivity)
  
  
  
#------------------------------------------------------------------------------#
#### Fit a Diffusion Model ####

# TODO: add Sebastians histogram


# prepare the data
dataset %>%
  # filter trials where the value difference is zero (since they can't be
  # allocated to either boundary)
  filter(! is.na(correct)) %>%
  # add a variable for the boundaries
  mutate(boundary = ifelse(correct == 1, "upper", "lower")) -> dataset


## returns the deviance of the four-parameter diffusion model 
## for given decisions (boundary), response times (RTs) and diffusion model
## parameters
diffusion_deviance <- function(boundary, RTs, v, a, z = .5 * a, t0) {
  eps <- .00000001
  likelihood <- ddiffusion(RTs, boundary, v = v, a = a, t0 = t0, z = z)
  # use a value very close to zero in case the returned likelihood is zero
  # (log(0) = -inf, leads to all kinds of bad things^^)
  likelihood[likelihood == 0] <- eps
  deviance <- - 2 * sum(log(likelihood))
  return(deviance)
}

# define arbitrary starting values
start_val <- c(0.3, 1, 0.5, 0.3)
# test if they lead to something sensible
diffusion_deviance(dataset$boundary, dataset$RT, 0.3, 1, 0.5, 0.3)

# fit the diffusion model to data of all participants
results_dm <- vector(mode = "list", length = N)
for (s_idx in 1:N) {
  subj_tmp <- subjects[[s_idx]]
  results_dm[[s_idx]] <- vector(mode = "list", length = 3)
  for (recall_idx in 1:length(unique(dataset$recallType))) {
    recall_tmp <- unique(dataset$recallType)[recall_idx]
    # filter the relevant data for the respective subject: the decisions
    # ("upper" for correct, "lower" for incorrect decisions) and response times
    boundaries <- dataset$boundary[dataset$subjID == subj_tmp & 
                                       dataset$recallType == recall_tmp]
    RTs <- dataset$RT[dataset$subjID == subj_tmp & dataset$recallType == recall_tmp]
    
    # fit the model using the nmk() algorithm
    fitModel <- nmk(par = start_val, fn = function(par) {
      diffusion_deviance(boundary = boundaries,
                         RTs = RTs,
                         v = par[1],
                         a = par[2],
                         z = par[3],
                         t0 = par[4])
    })
    # print(fitModel$message)
    # store the results
    results_dm[[s_idx]][[recall_idx]] <- fitModel
  }
}


