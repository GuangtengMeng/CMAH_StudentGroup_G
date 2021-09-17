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
dataset <- data_all %>% 
  mutate(RTs = RT / 1000) %>% 
  filter(!is.na(RT) & RTs>=0.150) %>% 
  filter(Cond == 'SSI' | Cond == 'SSC' | Cond == 'SRI' | Cond == 'SRC') %>% 
  mutate(boundary = ifelse(ACC == 1, "upper", "lower"))
  



# DDM ---------------------------------------------------------------------
#### Fit a Diffusion Model ####
# Have a look at the RT data / RT histograms
par(mfrow=c(2,2))
hist(dataset$RTs[(dataset$Cond=="SSI")&(dataset$ACC==1)],breaks = (0:60)/40,col = 'red',xlab = '',main = '',xaxt='n')
hist(dataset$RTs[(dataset$Cond=="SSI")&(dataset$ACC==0)],breaks = (0:60)/40,col = 'darkred',add = TRUE)
legend(2.5,250,c('SSI - Correct','SSI - Incorrect'),fill=c('red','darkred'),bty='n')


hist(dataset$RTs[(dataset$Cond=="SSC")&(dataset$ACC==1)],breaks = (0:60)/40,col = 'blue',xlab = '',main = '',xaxt='n')
hist(dataset$RTs[(dataset$Cond=="SSC")&(dataset$ACC==0)],breaks = (0:60)/40,col = 'darkblue',add = TRUE)
legend(2.5,250,c('SSC - Correct','SSC - Incorrect'),fill=c('blue','darkblue'),bty='n')

hist(dataset$RTs[(dataset$Cond=="SRI")&(dataset$ACC==1)],breaks = (0:60)/40,col = 'green',xlab = '',main = '',xaxt='n')
hist(dataset$RTs[(dataset$Cond=="SRI")&(dataset$ACC==0)],breaks = (0:60)/40,col = 'darkgreen',add = TRUE)
legend(2.5,250,c('SRI - Correct','SRI - Incorrect'),fill=c('green','darkgreen'),bty='n')


hist(dataset$RTs[(dataset$Cond=="SRC")&(dataset$ACC==1)],breaks = (0:60)/40,col = 'cyan',xlab = '',main = '',xaxt='n')
hist(dataset$RTs[(dataset$Cond=="SRC")&(dataset$ACC==0)],breaks = (0:60)/40,col = 'darkcyan',add = TRUE)
legend(2.5,250,c('SRC - Correct','SRC - Incorrect'),fill=c('cyan','darkcyan'),bty='n')

mtext("RT histograms per condition",side=3,line=-22,outer=TRUE)
ggsave("RT histograms per condition.png", width = 6, height = 4)

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
names(results_dm) <- c("Subject", "Cond", "drift_rate", "threshold", 
                       "starting_point", "non_decision_time", "deviance")
subjects <- unique(dataset$Subject)
conditions <- unique(dataset$Cond)

# loop through subjects + recall conditions
for (subj_tmp in 1:length(subjects)) {
  for (cond_tmp in 1:length(conditions)) {
    # filter the relevant data for the respective subject: the decisions
    # ("upper" for correct, "lower" for incorrect decisions) and response times
    boundaries <- dataset$boundary[dataset$Subject == subjects[subj_tmp] &
                                     dataset$Cond == conditions[cond_tmp]]
    RTs <- dataset$RTs[dataset$Subject == subjects[subj_tmp] &
                         dataset$Cond == conditions[cond_tmp]]
    
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


plot_dr <- results_dm %>% 
  filter(Subject != 115) %>% 
  

ggbetweenstats(plot_dr, x = Cond, y = drift_rate)
ggsave("drift rate.png", width = 6, height = 4)
ggbetweenstats(plot_dr, x = Cond, y = starting_point)
ggsave("starting_point.png", width = 6, height = 4)
ggbetweenstats(plot_dr, x = Cond, y = threshold)
ggsave("threshold.png", width = 6, height = 4)
ggbetweenstats(plot_dr, x = Cond, y = non_decision_time)
ggsave("non_decision_time.png", width = 6, height = 4)



# drift rate
results_dm %>%
  ggplot(aes(x = Cond, y = drift_rate, group = Cond, 
             color = Cond, fill = Cond)) +
  geom_jitter(alpha = .6, width = .13) + 
  geom_boxplot(width = .4, alpha = .5) + 
  guides(fill = "none", color = "none") +
  theme_minimal() +
  labs(y = "Drift Rate (v)", x = "Object")

ggsave("plots/drift rate.png", width = 6, height = 4)
ggbetweenstats(results_dm, x = Cond, y = drift_rate)
results_drift <- aov_ez(id = "Subject",
                        dv = "drift_rate",
                        data = results_dm,
                        within = c("Cond"),
                        anova_table = list(es = "pes"))
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


results_drift <- aov_ez(id = "Subject",
                        dv = "drift_rate",
                        data = results_dm,
                        within = c("Cond"),
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



