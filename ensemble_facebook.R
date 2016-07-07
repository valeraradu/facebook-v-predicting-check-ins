################### ensemble 2 submissions

# What the code basically does is create a weighted average of the places in the two submissions. Then it picks the top three places with the highest weights.

basedir = "/Users/valeriu/kaggle/facebook_check_ins"
setwd(basedir)
library('bit64')
library(data.table)
library(readr)

# These two lines give the relative weight of two submissions
weight_sub1 <- 0.5
weight_sub2 <- 0.5

# These three lines give the weight assigned to the first place listed in the submissions, the second place, and the third place
weight_place1 <- 1.0
weight_place2 <- 0.5
weight_place3 <- 0.333

# Read in data
submission1 <- as.data.frame(fread("submission_test$$_2016-06-09-13-58.csv"))
head(submission1)
nrow(submission1)
submission2 <- as.data.frame(fread("sub_knn.csv"))
head(submission2)
nrow(submission2)

# Parse out places into separate fields
submission1$place1   <- substr(submission1$place_id,1,regexpr(' ', submission1$place_id)-1)
submission1$place_id <- substr(submission1$place_id,regexpr(' ', submission1$place_id)+1,999)
submission1$place2   <- substr(submission1$place_id,1,regexpr(' ', submission1$place_id)-1)
submission1$place_id <- substr(submission1$place_id,regexpr(' ', submission1$place_id)+1,999)
submission1$place3   <- submission1$place_id
submission1$place_id <- NULL
submission1$weight1 <- weight_sub1 * weight_place1
submission1$weight2 <- weight_sub1 * weight_place2
submission1$weight3 <- weight_sub1 * weight_place3

submission2$place4   <- substr(submission2$place_id,1,regexpr(' ', submission2$place_id)-1)
submission2$place_id <- substr(submission2$place_id,regexpr(' ', submission2$place_id)+1,999)
submission2$place5   <- substr(submission2$place_id,1,regexpr(' ', submission2$place_id)-1)
submission2$place_id <- substr(submission2$place_id,regexpr(' ', submission2$place_id)+1,999)
submission2$place6   <- submission2$place_id
submission2$place_id <- NULL
submission2$weight4 <- weight_sub2 * weight_place1
submission2$weight5 <- weight_sub2 * weight_place2
submission2$weight6 <- weight_sub2 * weight_place3

head(submission1)
nrow(submission1)
head(submission2)
nrow(submission2)

# Merge into one data frame
submission <- merge(submission1,submission2)
head(submission)
nrow(submission)

# If any of the places in the first submission match places in the second submission, combine the weights and turn the place in the second submission to 'x'
submission$weight1 <- ifelse(submission$place1 == submission$place4,submission$weight1+submission$weight4,submission$weight1)
submission$place4  <- ifelse(submission$place1 == submission$place4,'x',submission$place4)
head(submission)
submission$weight2 <- ifelse(submission$place2 == submission$place4,submission$weight2+submission$weight4,submission$weight2)
submission$place4  <- ifelse(submission$place2 == submission$place4,'x',submission$place4)
head(submission)
submission$weight3 <- ifelse(submission$place3 == submission$place4,submission$weight3+submission$weight4,submission$weight3)
submission$place4  <- ifelse(submission$place3 == submission$place4,'x',submission$place4)
head(submission)

submission$weight1 <- ifelse(submission$place1 == submission$place5,submission$weight1+submission$weight5,submission$weight1)
submission$place5  <- ifelse(submission$place1 == submission$place5,'x',submission$place5)
head(submission)
submission$weight2 <- ifelse(submission$place2 == submission$place5,submission$weight2+submission$weight5,submission$weight2)
submission$place5  <- ifelse(submission$place2 == submission$place5,'x',submission$place5)
head(submission)
submission$weight3 <- ifelse(submission$place3 == submission$place5,submission$weight3+submission$weight5,submission$weight3)
submission$place5  <- ifelse(submission$place3 == submission$place5,'x',submission$place5)
head(submission)

submission$weight1 <- ifelse(submission$place1 == submission$place6,submission$weight1+submission$weight6,submission$weight1)
submission$place6  <- ifelse(submission$place1 == submission$place6,'x',submission$place6)
head(submission)
submission$weight2 <- ifelse(submission$place2 == submission$place6,submission$weight2+submission$weight6,submission$weight2)
submission$place6  <- ifelse(submission$place2 == submission$place6,'x',submission$place6)
head(submission)
submission$weight3 <- ifelse(submission$place3 == submission$place6,submission$weight3+submission$weight6,submission$weight3)
submission$place6  <- ifelse(submission$place3 == submission$place6,'x',submission$place6)
head(submission)

# Set weights to zero for all places that are now 'x'
submission$weight4 <- ifelse(submission$place4 == 'x',0,submission$weight4)
submission$weight5 <- ifelse(submission$place5 == 'x',0,submission$weight5)
submission$weight6 <- ifelse(submission$place6 == 'x',0,submission$weight6)

submission$place1a <- 'empty'
submission$place2a <- 'empty'
submission$place3a <- 'empty'

# Iteratively find maximim weight, use that place, remove weight for that place.  Repeat three times.
submission$maxweight <- pmax(submission$weight1,submission$weight2,submission$weight3,submission$weight4,submission$weight5,submission$weight6)


submission$place1a <- ifelse(submission$place1a == 'empty' & submission$weight1 == submission$maxweight,submission$place1,submission$place1a)
submission$maxweight <- ifelse(submission$place1a == submission$place1,-1.0,submission$maxweight)
submission$weight1 <- ifelse(submission$place1a == submission$place1,-1.0,submission$weight1)
submission$place1a <- ifelse(submission$place1a == 'empty' & submission$weight2 == submission$maxweight,submission$place2,submission$place1a)
submission$maxweight <- ifelse(submission$place1a == submission$place2,-1.0,submission$maxweight)
submission$weight2 <- ifelse(submission$place1a == submission$place2,-1.0,submission$weight2)
submission$place1a <- ifelse(submission$place1a == 'empty' & submission$weight3 == submission$maxweight,submission$place3,submission$place1a)
submission$maxweight <- ifelse(submission$place1a == submission$place3,-1.0,submission$maxweight)
submission$weight3 <- ifelse(submission$place1a == submission$place3,-1.0,submission$weight3)
submission$place1a <- ifelse(submission$place1a == 'empty' & submission$weight4 == submission$maxweight,submission$place4,submission$place1a)
submission$maxweight <- ifelse(submission$place1a == submission$place4,-1.0,submission$maxweight)
submission$weight4 <- ifelse(submission$place1a == submission$place4,-1.0,submission$weight4)
submission$place1a <- ifelse(submission$place1a == 'empty' & submission$weight5 == submission$maxweight,submission$place5,submission$place1a)
submission$maxweight <- ifelse(submission$place1a == submission$place5,-1.0,submission$maxweight)
submission$weight5 <- ifelse(submission$place1a == submission$place5,-1.0,submission$weight5)
submission$place1a <- ifelse(submission$place1a == 'empty' & submission$weight6 == submission$maxweight,submission$place6,submission$place1a)
submission$maxweight <- ifelse(submission$place1a == submission$place6,-1.0,submission$maxweight)
submission$weight6 <- ifelse(submission$place1a == submission$place6,-1.0,submission$weight6)

head(submission)
submission$maxweight <- pmax(submission$weight1,submission$weight2,submission$weight3,submission$weight4,submission$weight5,submission$weight6)

submission$place2a <- ifelse(submission$place2a == 'empty' & submission$weight1 == submission$maxweight,submission$place1,submission$place2a)
submission$maxweight <- ifelse(submission$place2a == submission$place1,-1.0,submission$maxweight)
submission$weight1 <- ifelse(submission$place2a == submission$place1,-1.0,submission$weight1)
submission$place2a <- ifelse(submission$place2a == 'empty' & submission$weight2 == submission$maxweight,submission$place2,submission$place2a)
submission$maxweight <- ifelse(submission$place2a == submission$place2,-1.0,submission$maxweight)
submission$weight2 <- ifelse(submission$place2a == submission$place2,-1.0,submission$weight2)
submission$place2a <- ifelse(submission$place2a == 'empty' & submission$weight3 == submission$maxweight,submission$place3,submission$place2a)
submission$maxweight <- ifelse(submission$place2a == submission$place3,-1.0,submission$maxweight)
submission$weight3 <- ifelse(submission$place2a == submission$place3,-1.0,submission$weight3)
submission$place2a <- ifelse(submission$place2a == 'empty' & submission$weight4 == submission$maxweight,submission$place4,submission$place2a)
submission$maxweight <- ifelse(submission$place2a == submission$place4,-1.0,submission$maxweight)
submission$weight4 <- ifelse(submission$place2a == submission$place4,-1.0,submission$weight4)
submission$place2a <- ifelse(submission$place2a == 'empty' & submission$weight5 == submission$maxweight,submission$place5,submission$place2a)
submission$maxweight <- ifelse(submission$place2a == submission$place5,-1.0,submission$maxweight)
submission$weight5 <- ifelse(submission$place2a == submission$place5,-1.0,submission$weight5)
submission$place2a <- ifelse(submission$place2a == 'empty' & submission$weight6 == submission$maxweight,submission$place6,submission$place2a)
submission$maxweight <- ifelse(submission$place2a == submission$place6,-1.0,submission$maxweight)
submission$weight6 <- ifelse(submission$place2a == submission$place6,-1.0,submission$weight6)

head(submission)
submission$maxweight <- pmax(submission$weight1,submission$weight2,submission$weight3,submission$weight4,submission$weight5,submission$weight6)

submission$place3a <- ifelse(submission$place3a == 'empty' & submission$weight1 == submission$maxweight,submission$place1,submission$place3a)
submission$maxweight <- ifelse(submission$place3a == submission$place1,-1.0,submission$maxweight)
submission$weight1 <- ifelse(submission$place3a == submission$place1,-1.0,submission$weight1)
submission$place3a <- ifelse(submission$place3a == 'empty' & submission$weight2 == submission$maxweight,submission$place2,submission$place3a)
submission$maxweight <- ifelse(submission$place3a == submission$place2,-1.0,submission$maxweight)
submission$weight2 <- ifelse(submission$place3a == submission$place2,-1.0,submission$weight2)
submission$place3a <- ifelse(submission$place3a == 'empty' & submission$weight3 == submission$maxweight,submission$place3,submission$place3a)
submission$maxweight <- ifelse(submission$place3a == submission$place3,-1.0,submission$maxweight)
submission$weight3 <- ifelse(submission$place3a == submission$place3,-1.0,submission$weight3)
submission$place3a <- ifelse(submission$place3a == 'empty' & submission$weight4 == submission$maxweight,submission$place4,submission$place3a)
submission$maxweight <- ifelse(submission$place3a == submission$place4,-1.0,submission$maxweight)
submission$weight4 <- ifelse(submission$place3a == submission$place4,-1.0,submission$weight4)
submission$place3a <- ifelse(submission$place3a == 'empty' & submission$weight5 == submission$maxweight,submission$place5,submission$place3a)
submission$maxweight <- ifelse(submission$place3a == submission$place5,-1.0,submission$maxweight)
submission$weight5 <- ifelse(submission$place3a == submission$place5,-1.0,submission$weight5)
submission$place3a <- ifelse(submission$place3a == 'empty' & submission$weight6 == submission$maxweight,submission$place6,submission$place3a)
submission$maxweight <- ifelse(submission$place3a == submission$place6,-1.0,submission$maxweight)
submission$weight6 <- ifelse(submission$place3a == submission$place6,-1.0,submission$weight6)

head(submission)

# Combine three places into space-separate string and write to CSV file.
submission$place_id <- paste(submission$place1a,submission$place2a,submission$place3a,sep=" ")

head(submission)

write.csv(submission[,c("row_id","place_id")],"submission_rf400-02-008_knn.csv", row.names = F)









