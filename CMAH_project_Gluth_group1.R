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
library(dfoptim)
library(rtdists)
library(afex)
library(RColorBrewer)
library(stats)


# import the data
dataset <- read.delim('data_allTrialTypes', header = TRUE, sep = ",")

set.seed(104234)

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
ggsave("plots/accuracy.png", width = 6, height = 4)


# same for the response times


dataset %>%
  group_by(subjID, recallType) %>%
  summarize(mean_rt = mean(RT)) -> data_aggregated_RT

data_aggregated_RT <- merge(data_aggregated_RT, data_aggregated_RT %>%
                              group_by(recallType) %>%
                              summarize(mean = mean(mean_rt),
                                        sd = sd(mean_rt)))

data_aggregated_RT %>%
  mutate(recallType = factor(recallType)) %>%
  ggplot(aes(x = recallType, y = mean_rt, color = recallType, fill = recallType)) +
  geom_jitter(width = .2, alpha = .8) +
  stat_summary(fun = mean, geom = "bar", width = .3, alpha = .3) +
  geom_errorbar(aes(ymin = mean - (1.98 * (sd / sqrt(N))),
                    ymax = mean + (1.98 * (sd / sqrt(N)))),
                width = .15, position = position_dodge(.9)) + 
  labs(x = "Number of options recalled", y = "Mean Individual Response Latencies") +
  theme_minimal() +
  guides(fill = F, color = F)
ggsave("plots/RTs.png", width = 6, height = 4)



#------------------------------------------------------------------------------#
#### Task 2: Fit a logit model to the data ####

#### simulate some data
steps <- seq(-3, 3, 1)

# returns the choice probability of the given value difference (valDiffSim)
p_right <- function(valDiffSim, rightBias = 1, choiceSensitivity = 1) {
  return(1 / (1 + exp(-rightBias - choiceSensitivity * valDiffSim)))
}

probabilities_sim <- p_right(valDiffSim = steps,
                             rightBias = -10,
                             choiceSensitivity = 1)
plot(x = steps, y = probabilities_sim)


#### build a likelihood function

# returns the negative log likelihood for a given value difference and choice
# according to the logit model
choiceModel1 <- function(valDiff, choiceRight, recall_condition, 
                         rightBias = 1, choiceSensitivity = c(1, 1, 1)) {
  # print("Hello???... It's me.")
  choiceSensitivity_vector <- choiceSensitivity[1] * (recall_condition == 0) +
    choiceSensitivity[2] * (recall_condition == 1) +
    choiceSensitivity[3] * (recall_condition == 2)
  p_right <- 1 / (1 + exp(-rightBias - choiceSensitivity_vector * valDiff))
  neg_log_lik <- - (sum(log(p_right[choiceRight == 1])) + 
    sum(log(1 - p_right[choiceRight == 0])))
  return(neg_log_lik)
}


#### Fit the logit model using grid search ####

# initialize some values 
bias <- seq(-2, 2, .1)
choiceSensitivity <- seq(-2, 2, .1)
subjects <- unique(dataset$subjID)
best_bias <- rep(NA, N)
best_sensitivity <- rep(NA, N)
best_nLL <- rep(1/0, N)

# fit the model separately to all subjects by trying different values for 
# both parameters
for (s_idx in 1:N) {
  subj_tmp <- subjects[s_idx]
  for (bias_idx in 1:length(bias)) {
    bias_tmp <- bias[bias_idx]
    for (choice_sens_idx in 1:length(choiceSensitivity)) {
      choice_sens_tmp <- choiceSensitivity[choice_sens_idx]
      choiceRight <- dataset$choiceRight[which(dataset$subjID == subj_tmp)]
      valDiff <- dataset$valDiff[which(dataset$subjID == subj_tmp)]
      # compute the negative log likelihood of the parameter values in the current
      # iteration for subejct subj_tmp
      neg_log_lik <- choiceModel1(valDiff = valDiff,
                                  choiceRight = choiceRight,
                                  rightBias = bias_tmp,
                                  choiceSensitivity = rep(choice_sens_tmp, 3),
                                  recall_condition = rep(0, length(valDiff))
                                  )
      # update the best parameter values if the values return a smaller (i.e., better)
      # negative log likelihood
      if (neg_log_lik < best_nLL[s_idx]) {
        best_bias[s_idx] <- bias_tmp
        best_sensitivity[s_idx] <- choice_sens_tmp
        best_nLL[s_idx] <- neg_log_lik
      }
    }
  }
}


#### Fit the logit model using an optimization algorithm (i.e., Nelder-Mead optimization) ####
results_logit <- vector(mode = "list", length = N)
for (s_idx in 1:N) {
  subj_tmp <- subjects[s_idx]
  startVal <- c(best_bias[s_idx], best_sensitivity[s_idx])
  choiceRight <- dataset$choiceRight[dataset$subjID == subj_tmp]
  valDiff <- dataset$valDiff[dataset$subjID == subj_tmp]
  fitModel <- optim(par = startVal, fn = function(par) {
    choiceModel1(rightBias = par[1],
                 choiceSensitivity = rep(par[2], 3),
                 choiceRight = choiceRight,
                 valDiff = valDiff,
                 # give a vector with only one element because we don't take into
                 # account the recall condition
                 recall_condition = rep(0, length(valDiff)))
    })
  results_logit[[s_idx]] <- fitModel
}


# test whether the built-in glm() function leads to the same results for the
# first subjects
data_test <- dataset[dataset$subjID == 1, ]
model_test <- glm(choiceRight ~ valDiff, family = binomial(link = "logit"),
                  data = data_test)

# it does, yay :)


#------------------------------------------------------------------------------#
#### Fit the logit model (separate sensitivity parameters) ####

results_sensitivity <- data.frame(matrix(nrow = 0, ncol = 6))
names(results_sensitivity) <- c("subject", "choiceRight", "sensitivity_0", "sensitivity_1",
                    "sensitivity_2", "deviance")
startVal <- c(0.5, 0.5, 0.5, 0.5)

for (s_idx in 1:N) {
  # print("hi")
  subj_tmp <- subjects[[s_idx]]
  # startVal <- c(best_bias[s_idx], rep(best_sensitivity[s_idx], 3))
  choiceRight <- dataset$choiceRight[which(dataset$subjID == subj_tmp)]
  valDiff <- dataset$valDiff[dataset$subjID == subj_tmp]
  recallType <- dataset$recallType[dataset$subjID == subj_tmp]
  fitModel <- nmk(par = startVal, fn = function(par) {
    choiceModel1(rightBias = par[1],                   
                 choiceSensitivity = par[2:4],
                 choiceRight = choiceRight,
                 valDiff = valDiff,
                 recall_condition = recallType)
  })
  results_tmp <- list(subj_tmp, fitModel$par[1], fitModel$par[2], fitModel$par[3],
                      fitModel$par[4], fitModel$value)
  names(results_tmp) <- names(results_sensitivity)
  results_sensitivity <- rbind(results_sensitivity, results_tmp)
}


#### Plot Results for Sensitivity Model ####

results_sensitivity %>%
  dplyr::select(-deviance) %>%
  pivot_longer(cols = c(sensitivity_0, sensitivity_1, sensitivity_2),
               names_to = "recall_condition", values_to = "parameter") %>%
  mutate(recall_condition = str_sub(recall_condition, -1, -1)) %>%
  ggplot(aes(x = recall_condition, y = parameter, group = recall_condition,
             color = recall_condition, fill = recall_condition)) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  theme_minimal() +
  guides(fill = F, color = F) +
  ylim(-1.5, 2) +
  labs(y = "Sensitivity (Logit Model)", x = "Number of Options Recalled") +
  guides(fill = F, color = F)
ggsave("plots/sensitivity_params.png", width = 6, height = 4)



#------------------------------------------------------------------------------#
#### Fit a Diffusion Model to Choices + RTs ####

# prepare the data
dataset %>%
  # filter trials where the value difference is zero (since they can't be
  # allocated to either boundary)
  filter(! is.na(correct)) %>%
  # add a variable for the boundaries
  mutate(boundary = ifelse(correct == 1, "upper", "lower")) -> dataset


## Returns the deviance of the four-parameter diffusion model 
## for given decisions (boundary), response times (RTs) and diffusion model
## parameters (v, a, z, a, t0)
diffusion_deviance <- function(boundary, RTs, v, a, z = .5 * a, t0) {
  likelihood <- ddiffusion(RTs, boundary, v = v, a = a, t0 = t0, z = z)
  # use a value very close to zero in case the returned likelihood is zero
  # (log(0) = -inf, leads to all kinds of bad things)
  eps <- .00000001
  likelihood[likelihood == 0] <- eps
  deviance <- - 2 * sum(log(likelihood))
  return(deviance)
}

# define arbitrary starting values
start_val <- c(0.3, 1, 0.5, 0.3)
# test if they lead to something sensible
diffusion_deviance(dataset$boundary, dataset$RT, 0.3, 1, 0.5, 0.3)
# not Inf or -Inf --> ok

##### fit the diffusion model to data of all participants

# initialize the df to store results
results_dm <- data.frame(matrix(nrow = 0, ncol = 7))
names(results_dm) <- c("subject", "recall_condition", "drift", "threshold", 
                       "starting_point", "non_decision_time", "deviance")
subjects <- unique(dataset$subjID)
recall_type <- unique(dataset$recallType)

# loop through subjects + recall conditions
for (subj_tmp in subjects) {
  for (recall_tmp in recall_type) {
    # filter the relevant data for the respective subject: the decisions
    # ("upper" for correct, "lower" for incorrect decisions) and response times
    boundaries <- dataset$boundary[dataset$subjID == subj_tmp & 
                                       dataset$recallType == recall_tmp]
    RTs <- dataset$RT[dataset$subjID == subj_tmp & 
                        dataset$recallType == recall_tmp]
    
    # fit the model using the Nelder-Mead algorithm
    fitModel <- nmk(par = start_val, fn = function(par) {
      diffusion_deviance(boundary = boundaries,
                         RTs = RTs,
                         v = par[1],
                         a = par[2],
                         z = par[3],
                         t0 = par[4])
    })
    # store results
    results_tmp <- list(subj_tmp, recall_tmp, fitModel$par[1], fitModel$par[2],
                       fitModel$par[3], fitModel$par[4], fitModel$value)
    names(results_tmp) <- names(results_dm)
    results_dm <- rbind(results_dm, results_tmp)
  }
}

# compute the relative starting point -> relative to the decision threshold
results_dm %>% 
  mutate(starting_point_relative = starting_point / threshold,
         recall_condition = factor(recall_condition)) -> results_dm


#------------------------------------------------------------------------------#
#### Plot DM parameters ####

# drift rate
results_dm %>%
  ggplot(aes(x = recall_condition, y = drift, group = recall_condition, 
             color = recall_condition, fill = recall_condition)) +
  # geom_violin(alpha = .4) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  # geom_line(aes(y = mean(drift))) +
  labs(y = "Drift Rate (v)", x = "Number of Options Recalled") +
  theme_minimal() +
  guides(fill = F, color = F)
ggsave("plots/drift.png", width = 6, height = 4)
# descriptive difference in drift rate

# starting point
results_dm %>%
  ggplot(aes(x = recall_condition, y = starting_point / threshold, group = recall_condition, 
             color = recall_condition, fill = recall_condition)) +
  # geom_violin(alpha = .4) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  theme_minimal() +
  labs(y = "Starting Point (z, relative to the threshold)", x = "Number of Options Recalled") +
  guides(fill = F, color = F)
ggsave("plots/starting_point.png", width = 6, height = 4)

# threshold
results_dm %>%
  ggplot(aes(x = recall_condition, y = threshold, group = recall_condition, 
             color = recall_condition, fill = recall_condition)) +
  # geom_violin(alpha = .4) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  theme_minimal() +
  labs(y = "Threshold (a)", x = "Number of Options Recalled") +
  guides(fill = F, color = F)
ggsave("plots/threshold.png", width = 6, height = 4)

# non-decision time
results_dm %>%
  ggplot(aes(x = recall_condition, y = non_decision_time, group = recall_condition, 
             color = recall_condition, fill = recall_condition)) +
  # geom_violin(alpha = .4) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  theme_minimal() +
  labs(y = "Non-Decision Time (t0)", x = "Number of Options Recalled") +
  guides(fill = F, color = F)
ggsave("plots/non_decision_time.png", width = 6, height = 4)


results_drift <- aov_ez(id = "subject",
       dv = "drift",
       data = results_dm,
       within = c("recall_condition"),
       anova_table = list(es = "pes"))
# p < .001 (very significant^^)

results_starting_point <- aov_ez(id = "subject",
                                 dv = "starting_point_relative",
                                 data = results_dm,
                                 within = c("recall_condition"),
                                 anova_table = list(es = "pes"))
# n.s.

results_threshold <- aov_ez(id = "subject",
                                 dv = "threshold",
                                 data = results_dm,
                                 within = c("recall_condition"),
                                anova_table = list(es = "pes"))
# p < .001 significantly significant^^

results_non_decision_time <- aov_ez(id = "subject",
                            dv = "non_decision_time",
                            data = results_dm,
                            within = c("recall_condition"),
                            anova_table = list(es = "pes"))
# p = .028 -> slightly significant



