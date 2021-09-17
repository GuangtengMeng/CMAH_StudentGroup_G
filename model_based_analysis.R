###########################
### Carelab - Empathy #
###########################
### Analysis code #
####################################
### menggt@psych.ac.cn #
####################################

## Background information ##
##
##
## in this study, participants worked on the so-called pain-and-share task:
## this is a block designed experiment, in the very first of every block, people are informed with their partner during this block
## in every trial, people first receive the stimulation and evaluate the pain level of themselves and others
## then their partners are going to receive a second stimulation, people can choose to share the probability of pain or not
## one of the participants and their partners will receive the stimulation, and participants will evaluate the pain level again
## this procedure is repeated 4 blocks (2 for friends and 2 for strangers),
## 6 conditions repeated 5 times respectively (totally 30 times every blocks)

## Goals of this analysis ##
## 
## 1. merge the data files
## 2. get a feeling for the data by analyzing the acceptance as a function of whether friend or stranger was concerned
## 2. fit a logit model (McFadden, 2001, American Economic Review) to the decisions, taking the pain ratings into account
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

# import library ---------------------------------------------------------
library(dplyr)   # CRAN v1.0.7
library(tidyverse) # Easily Install and Load the 'Tidyverse'
library(ggplot2) # Create Elegant Data Visualisations Using the Grammar of Graphics
library(dfoptim) # Derivative-Free Optimization
library(rtdists) # Response Time Distributions
library(ggstatsplot) # 'ggplot2' Based Plots with Statistical Details
library(afex) # not installed on this machine




# merge data files --------------------------------------------------------
data_all <- list.files(path = dirname(rstudioapi::getActiveDocumentContext()$path),
                       pattern = "*.csv",
                       full.names = TRUE) %>% 
  lapply(read.csv) %>% 
  bind_rows

# check data
dim(data_all)
head(data_all)
tail(data_all)

# filter
## check numbers of object conditions
data_all %>% filter(!is.na(intensity) | !is.na(proportion)) %>% 
  count(object)

## output
dataset <- data_all %>% 
  select(participant, gender, object, intensity, proportion, 
                               sliderSelfBS.response, krSelfBS.rt, sliderOtherBS.response, krOtherBS.rt,
                               krAccept.keys, krAccept.corr, krAccept.rt, shareshock_target,
                               sliderAS.response, krAS.rt,  ll, ul) %>% 
  filter(!is.na(intensity) | !is.na(intensity))

# Right choice & rating difference -------------------------------------------------------
dataset$choiceRight <- ifelse(dataset$krAccept.keys == "right", 1, 0)
dataset$shockTarget <- ifelse(dataset$krAccept.corr == 1 & dataset$shareshock_target == 0, "self", dataset$object)
dataset$ratingDiff <- dataset$sliderOtherBS.response -dataset$sliderSelfBS.response

## save .csv file named by participants
write.csv(dataset, file = "data/01_dataset_001_002.csv")


# read file ---------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dataset <- read.csv("data/01_dataset_001_002.csv")

N <- length(unique(dataset$participant))

sum_dataset <- dataset %>%
  group_by(participant, object) %>% 
  summarise(acceptance = mean(krAccept.corr))

sum_dataset <- merge(sum_dataset,
                     sum_dataset %>% 
                       group_by(object) %>% 
                       summarise(mean_acceptance = mean(acceptance),
                                 sd_acceptance = sd(acceptance)),
                     by = "object")

sum_dataset %>% 
  ggplot(aes(x = object, y = acceptance, color = object, fill = object)) +
  # display the mean accuracy in each condition with a bar
  stat_summary(fun = mean, geom = "bar", width = .3, alpha = .3) +
  # also display individual datapoints
  geom_jitter(width = .2, alpha = .8) +
  # make errorbars with confidence intervals
  geom_errorbar(aes(ymin = mean_acceptance - (sd_acceptance / sqrt(N)),
                    ymax = mean_acceptance + (sd_acceptance / sqrt(N))),
                width = .15, position = position_dodge(.9)) + 
  # hide the legend
  guides(color = "none", fill = "none") +
  labs(x = "Object", y = "Acceptance of Share") +
  theme_minimal()
ggsave("plots/acceptance.png", width = 6, height = 4)

df_anova <- list(dataset$sliderSelfBS.response[dataset$shockTarget == "self"],
             array_2 = dataset$sliderAS.response[dataset$shockTarget == "self"],
             array_3 = dataset$sliderOtherBS.response[dataset$shockTarget != "self"],
             array_4 = dataset$sliderAS.response[dataset$shockTarget != "self"])

# Fit a logit model -------------------------------------------------------
p_right <- function(ratingDiff, rightBias = 1, choiceSensitivity = 1) {
  return(1 / (1 + exp(-rightBias - choiceSensitivity * ratingDiff)))
}

choiceModel1 <- function(ratingDiff, choiceRight, rightBias = 1, choiceSensitivity = 1){
  p_right <- 1./(1 + exp(-rightBias - choiceSensitivity * ratingDiff))
  #minimize the summed negative log-likelihood
  likelihood <- p_right * (choiceRight == 1) + (1-p_right) * (choiceRight == 0) #likelihood of observed behavior given the model
  negLogLikelihood <- -sum(log(likelihood)) #the negative log-likelihood is the target of the minimization algorithm
  
  return(negLogLikelihood)
}


bias <- seq(-2, 2, .1)
choiceSensitivity <- seq(-2, 2, .1)
subjects <- unique(dataset$participant)
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
      choiceRight <- dataset$choiceRight[dataset$participant == subj_tmp]
      ratingDiff <- dataset$ratingDiff[dataset$participant == subj_tmp]
      
      # compute the negative log likelihood of the parameter values in the current
      # iteration for subj_tmp
      neg_log_lik <- choiceModel1(ratingDiff = ratingDiff, choiceRight = choiceRight,
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

#### Fit the model using an optimization algorithm (i.e., Nelder-Mead optimization)
# nmk_results <- vector("list", length = N)
bestParamsnmkSearch <- matrix(data = 1/0, nrow = length(subjects), ncol = 3)

for (s_idx in 1:length(subjects)) {
  subj_tmp <- subjects[s_idx]
  
  # starting value for two parameters
  startVal <- c(bestParamsGridSearch[s_idx, 1], bestParamsGridSearch[s_idx, 2])
  
  # feed data
  choiceRight <- dataset$choiceRight[dataset$participant == subj_tmp]
  ratingDiff <- dataset$ratingDiff[dataset$participant == subj_tmp]
  
  # use nmk function of R-package dfoptim
  fitModel <- nmk(par = startVal, fn = function(par) {
    choiceModel1(rightBias = par[1], choiceSensitivity = par[2],
                 choiceRight = choiceRight, ratingDiff = ratingDiff)
  })
  
  # save outputs and print convergence messages
  bestParamsnmkSearch[s_idx, 1] <- fitModel$par[1]
  bestParamsnmkSearch[s_idx, 2] <- fitModel$par[2]
  bestParamsnmkSearch[s_idx, 3] <- fitModel$value
  
  print(fitModel$message)
}


# rightBiase
t.test(bestParamsnmkSearch[,1] - 0)

# choiceSensitivity
t.test(bestParamsnmkSearch[,2] - 0)


choiceModel2 <- function(ratingDiff, choiceRight, recall_condition, 
                         rightBias = 1, choiceSensitivity = c(1, 1)) {
  choiceSensitivity_vector <- choiceSensitivity[1] * (object=="partner")+
    choiceSensitivity[2] * (recallType=="stranger")
  
  p_right <- 1 / (1 + exp(-rightBias - choiceSensitivity_vector * ratingDiff))
  neg_log_lik <- - (sum(log(p_right[choiceRight == 1])) + 
                      sum(log(1 - p_right[choiceRight == 0])))
  return(neg_log_lik)
}


#### Assume that the choiceSensitivity parameter differs across conditions ####
bestParamsnmkSearch_conditions <- matrix(data = 1/0, nrow = length(subjects), ncol = 4)


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

#### Have a look at the parameter estimates
df_data_plot <- data.frame(object = c("partner", "stranger"),
                           choiceSensitivity = c(bestParamsnmkSearch[,1:2]))

plot(x = df_data_plot$object, y = df_data_plot$choiceSensitivity, ylim = c(-0.2,1.0))

df_data_plot <- merge(df_data_plot,
                      df_data_plot %>% 
                        group_by(object) %>% 
                        summarise(mean_choiceSensitivity = mean(choiceSensitivity),
                                  sd_choiceSensitivity = sd(choiceSensitivity)),
                      by = "object")

df_data_plot %>% 
  ggplot(aes(x = object, y = choiceSensitivity, color = object, fill = object)) +
  # display the mean accuracy in each condition with a bar
  stat_summary(fun = mean, geom = "bar", width = .3, alpha = .3) +
  # also display individual datapoints
  geom_jitter(width = .2, alpha = .8) +
  # make errorbars with confidence intervals
  geom_errorbar(aes(ymin = mean_choiceSensitivity - (sd_choiceSensitivity / sqrt(N)),
                    ymax = mean_choiceSensitivity + (sd_choiceSensitivity / sqrt(N))),
                width = .15, position = position_dodge(.9)) + 
  # hide the legend
  guides(color = "none", fill = "none") +
  labs(x = "Object", y = "Choice Sensitivity based on logit model") +
  theme_minimal()

#### Fit a Diffusion Model ####
# Have a look at the RT data / RT histograms
par(fig=c(0,1,0.3,1))
hist(dataset$krAccept.rt[(dataset$object=="partner")&(dataset$krAccept.corr==1)],breaks = (0:50)/10,col = 'red',xlab = '',main = '',xaxt='n')
hist(dataset$krAccept.rt[(dataset$object=="partner")&(dataset$krAccept.corr==0)],breaks = (0:50)/10,col = 'darkred',add = TRUE)
legend(2.5,250,c('partner - accept','partner - reject'),fill=c('red','darkred'),bty='n')
title('RT histograms per condition')
# par(fig=c(0,1,0.25,0.75),new=TRUE)
# hist(dataset$RT[(dataset$recallType==1)&(dataset$accuracy==1)],breaks = (0:50)/10,col = 'green',xlab = '',main = '',xaxt='n')
# hist(dataset$RT[(dataset$recallType==1)&(dataset$accuracy==0)],breaks = (0:50)/10,col = 'darkgreen',add = TRUE)
# legend(2.5,400,c('one remembered - correct','one remembered - error'),fill=c('green','darkgreen'),bty='n')
par(fig=c(0,1,0,0.7),new=TRUE)
hist(dataset$krAccept.rt[(dataset$object=="stranger")&(dataset$krAccept.corr==1)],breaks = (0:50)/10,col = 'blue',xlab = 'RT in sec',main = '')
hist(dataset$krAccept.rt[(dataset$object=="stranger")&(dataset$krAccept.corr==0)],breaks = (0:50)/10,col = 'darkblue',add = TRUE)
legend(2.5,275,c('both remembered - correct','both remembered - error'),fill=c('blue','darkblue'),bty='n')


# prepare the data
ddm_dataset <- dataset %>% 
  # filter trials where the value difference is zero (since they can't be
  # allocated to either boundary)
  filter(!is.na(krAccept.rt)) %>% 
  mutate(boundary = ifelse(correct == 1, "upper", "lower")) -> ddm_dataset

## returns the deviance of the four-parameter diffusion model 
## for given decisions (boundary), response times (RTs) and diffusion model
## parameters
diffusion_deviance <- function(RTs, boundary, v, a, z = .5 * a, t0) {
  likelihood <- ddiffusion(RTs, boundary, v = v, a = a, t0 = t0, z = z)
  # use a value very close to zero in case the returned likelihood is zero
  # (log(0) = -inf, leads to all kinds of bad things^^)
  likelihood[likelihood == 0] <- .00000001
  deviance <- - 2 * sum(log(likelihood))
  
  return(deviance)
}

# define arbitrary starting values
start_val <- c(0.3, 1, 0.5, 0.3)
# test if they lead to something sensible
diffusion_deviance(ddm_dataset$krAccept.rt, ddm_dataset$boundary, 0.3, 1, 0.5, 0.3)


##### fit the diffusion model to data of all participants

# initialize the df to store results
results_dm <- data.frame(matrix(nrow = 0, ncol = 7))
names(results_dm) <- c("participant", "object", "drift_rate", "threshold", 
                       "starting_point", "non_decision_time", "deviance")
subjects <- unique(dataset$participant)
conditions <- unique(dataset$object)

# loop through subjects + recall conditions
for (subj_tmp in 1:length(subjects)) {
  for (cond_tmp in 1:length(conditions)) {
    # filter the relevant data for the respective subject: the decisions
    # ("upper" for correct, "lower" for incorrect decisions) and response times
    boundaries <- ddm_dataset$boundary[ddm_dataset$participant == subjects[subj_tmp] &
                                         ddm_dataset$object == conditions[cond_tmp]]
    RTs <- ddm_dataset$krAccept.rt[ddm_dataset$participant == subjects[subj_tmp] &
                                     ddm_dataset$object == conditions[cond_tmp]]
    
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
    results_tmp <- list(subjects[subj_tmp], conditions[cond_tmp], fitModel$par[1], fitModel$par[2],
                        fitModel$par[3], fitModel$par[4], fitModel$value)
    names(results_tmp) <- names(results_dm)
    results_dm <- rbind(results_dm, results_tmp)
  }
}

# compute the relative starting point -> relative to the decision threshold
# results_dm %>% 
#   mutate(starting_point_relative = starting_point / threshold,
#          object_condition = factor(object)) -> results_dm


#------------------------------------------------------------------------------#
#### Plot DM parameters ####
plot_list <- names(results_dm)[3:6]
for (n_plot in 1:length(plot_list)) {
  results_dm %>% 
    ggbetweenstats(x = object,
                   y = 3,
                   xlab = "Object",
                   ylab = plot_list[n_plot],
                   title = "Estimated Drift Rates with Diffusion Model Analysis",
                   effsize.type = "eta")
}


# drift rate
results_dm %>%
  ggplot(aes(x = object, y = drift_rate, group = object, 
             color = object, fill = object)) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  guides(fill = "none", color = "none") +
  theme_minimal() +
  labs(y = "Drift Rate (v)", x = "Object")

ggsave("plots/drift rate.png", width = 6, height = 4)
# descriptive difference in drift rate

# starting point
results_dm %>%
  ggplot(aes(x = object, y = starting_point, group = object, 
             color = object, fill = object)) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  guides(fill = "none", color = "none") +
  theme_minimal() +
  labs(y = "Starting Point", x = "Object")

ggsave("plots/starting_point.png", width = 6, height = 4)

# threshold
results_dm %>%
  ggplot(aes(x = object, y = threshold, group = object, 
             color = object, fill = object)) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  guides(fill = "none", color = "none") +
  theme_minimal() +
  labs(y = "Threshold", x = "Object")

ggsave("plots/threshold.png", width = 6, height = 4)

# non-decision time
results_dm %>%
  ggplot(aes(x = object, y = non_decision_time, group = object, 
             color = object, fill = object)) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  guides(fill = "none", color = "none") +
  theme_minimal() +
  labs(y = "Non-Decision Time", x = "Object")

ggsave("plots/non_decision_time.png", width = 6, height = 4)


results_drift <- aov_ez(id = "participant",
                        dv = "drift_rate",
                        data = results_dm,
                        within = c("object"),
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



