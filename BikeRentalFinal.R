#Clearing RAM
rm(list = ls())

#importing all the required libraries
library(rpart)
library(corrplot)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(tidyverse)
library(modelr)
library(broom)
library(MLmetrics)
library(randomForest)
library(ggplot2)
library(caret)
library(e1071)

#Knowing the working directory
getwd()

#setting the working directory
setwd("C:/Users/Harish/Desktop/Projects")

#Importing the data
ddata  = read.csv("day.csv",sep = ",")

#Understanding the data or summary of the day data

head(ddata,5)
summary(ddata)
View(ddata)

#Knowing the data type of the variables
str(ddata)

#Data pre processing
#Cheking for any missing values in the dataset
sum(is.na(ddata))

#No missing values in our data, so no need of Imputing
#Understanding the Distribution of numeric variables by plotting Histogram

par(mfrow=c(4,2))
par(mar = rep(2, 4))
hist(ddata$season)
hist(ddata$weathersit)
hist(ddata$hum)
hist(ddata$holiday)
hist(ddata$workingday)
hist(ddata$temp)
hist(ddata$atemp)
hist(ddata$windspeed)

#outlier analysis
#Boxplotting count againest required variables to know the outliers

boxplot(ddata$registered~ddata$season,xlab="season", ylab="registered users")
boxplot(ddata$casual~ddata$season,xlab="season", ylab="casual users")

boxplot(ddata$registered~ddata$weathersit,xlab="weather", ylab="registered users")
boxplot(ddata$casual~ddata$weathersit,xlab="weather", ylab="casual users")

boxplot(ddata$registered~ddata$temp,xlab="temparature", ylab="registered users")
boxplot(ddata$casual~ddata$temp,xlab="temparature", ylab="casual users")

boxplot(ddata$registered~ddata$yr,xlab="year", ylab="registered users")
boxplot(ddata$casual~ddata$yr,xlab="year", ylab="casual users")

boxplot(ddata$registered~ddata$windspeed,xlab="windspeed", ylab="registered users")
boxplot(ddata$casual~ddata$windspeed,xlab="windspeed", ylab="casual users")

boxplot(ddata$registered~ddata$hum,xlab="humidity", ylab="registered users")
boxplot(ddata$casual~ddata$hum,xlab="humidity", ylab="casual users")

#From the Boxplots we can see there are outliers, 
#since the users are not normally or functionally distributed
#There is no need to take any action on these outliers

#Correlation Analysis
#Plotting a correlation heatmap

d = data.frame( ddata$temp,ddata$atemp,ddata$hum,ddata$windspeed,
                ddata$casual,ddata$registered,ddata$cnt)
M = cor(d)
corrplot(M, method = "circle")

#from the correlation plot it is clear that there is high correlation
#betwwen temp and atemp, so we can use any one while building the model

#plotting the random tree partitioning with data of users against temparature

f=rpart(registered~temp,ddata)
fancyRpartPlot(f)

f1=rpart(casual~temp,ddata)
fancyRpartPlot(f1)

#Applying feature selection
#Most of the available data is processed, let us create new
#Variables for better data feeding
#Feature Engineering the data to create some more meaningful columns

ddata$weekend=0
ddata$weekend[ddata$weekday== 0 | ddata$weekday== 6 ]=1

ddata$temp_reg=0
ddata$temp_reg[ddata$temp<13]=1
ddata$temp_reg[ddata$temp>=13 & ddata$temp<23]=2
ddata$temp_reg[ddata$temp>=23 & ddata$temp<30]=3
ddata$temp_reg[ddata$temp>=30]=4

ddata$temp_cas=0
ddata$temp_cas[ddata$temp<15]=1
ddata$temp_cas[ddata$temp>=15 & ddata$temp<23]=2
ddata$temp_cas[ddata$temp>=23 & ddata$temp<30]=3
ddata$temp_cas[ddata$temp>=30]=4


ddata$year_part[ddata$yr== 0 ]=1
ddata$year_part[ddata$yr== 0 & ddata$mnth>3]=2
ddata$year_part[ddata$yr== 0 & ddata$mnth>6]=3
ddata$year_part[ddata$yr== 0 & ddata$mnth>9]=4
ddata$year_part[ddata$yr== 1]=5
ddata$year_part[ddata$yr== 1 & ddata$mnth>3]=6
ddata$year_part[ddata$yr== 1 & ddata$mnth>6]=7
ddata$year_part[ddata$yr== 1 & ddata$mnth>9]=8
table(ddata$year_part)

ddata$day_type=0
ddata$day_type[ddata$holiday==0 & ddata$workingday==0]="weekend"
ddata$day_type[ddata$holiday==1]="holiday"
ddata$day_type[ddata$holiday==0 & ddata$workingday==1]="working day"
table(ddata$day_type)

#knowing the datatype of all available variables
str(ddata)

#Converting the variables to the required data format for feeding to model
ddata$season        =as.factor(ddata$season)
ddata$yr            =as.factor(ddata$yr)
ddata$weekday       =as.factor(ddata$weekday)
ddata$holiday       =as.factor(ddata$holiday)
ddata$workingday    =as.factor(ddata$workingday)
ddata$weekend       =as.factor(ddata$weekend)
ddata$weathersit    =as.factor(ddata$weathersit)
ddata$temp_cas      =as.factor(ddata$temp_cas) 
ddata$temp_reg      =as.factor(ddata$temp_reg) 
ddata$mnth          =as.factor(ddata$mnth)
ddata$day_type      =as.factor(ddata$day_type)
ddata$dteday        =as.factor(ddata$dteday)
ddata$year_part     =as.factor(ddata$year_part)

#data pre processing completed 
#Model building to predict the registered and casual users
#dividing the train and test data

dtrain =ddata[as.integer(substr(ddata$dteday,9,10))<21,]
dtest  =ddata[as.integer(substr(ddata$dteday,9,10))>20,]


#registered and casual users count is not normally distributed
#applying log transformation to these skewed variables in the data

dtrain$logcasual   =log(dtrain$casual+1)
dtrain$logregister =log(dtrain$registered+1)

#we divided the train and test data
#building the models with train data
#For first building i am going with linear regression model

lm_register = lm(logregister ~ workingday+holiday+day_type+hum+atemp+windspeed+season+weathersit+weekend+yr+year_part+mnth+day_type,data=dtrain)

lm_casual   = lm(logcasual ~ workingday+holiday+day_type+hum+atemp+windspeed+season+weathersit+weekend+yr+year_part+mnth+day_type,data=dtrain)

summary(lm_casual)

summary(lm_register)

Prediction_lmreg = predict(lm_register,dtest)
dtest$logregister= Prediction_lmreg

Prediction_lmcas = predict(lm_casual,dtest)
dtest$logcasual  = Prediction_lmcas

dtest$pregistered =exp(dtest$logregister)-1
dtest$pcasual     =exp(dtest$logcasual)-1

MAPE(dtest$pregistered,dtest$registered)

#errorrate = 7.9
#accuracy = 92.1

MAPE(dtest$pcasual,dtest$casual)

#errorrate = 8.4
#accuracy = 91.6

#Now Building model with random forest

set.seed(415)
rf_register = randomForest(logregister ~ workingday+holiday+day_type+hum+atemp+windspeed+season+weathersit+weekend+yr+year_part+mnth+day_type, data=dtrain,importance=TRUE, ntree=250)

set.seed(415)
rf_casual   = randomForest(logcasual ~ workingday+holiday+day_type+hum+atemp+windspeed+season+weathersit+weekend+yr+year_part+mnth+day_type, data=dtrain,importance=TRUE, ntree=250)


Prediction_rfreg = predict(rf_register,dtest)
dtest$rflogregister= Prediction_rfreg

Prediction_rfcas = predict(rf_casual,dtest)
dtest$rflogcasual  = Prediction_rfcas

dtest$rfpregistered =exp(dtest$rflogregister)-1
dtest$rfpcasual     =exp(dtest$rflogcasual)-1

MAPE(dtest$rfpregistered,dtest$registered)

#errorrate = 11.3
#accuracy = 88.7

MAPE(dtest$rfpcasual,dtest$casual)

#errorrate = 10
#accuracy = 90


dtest$fcnt=dtest$pcasual+dtest$pregistered


#Writing the data to the final output file

s = data.frame(day=dtest$dteday,registered=dtest$pregistered,casual=dtest$pcasual,count=dtest$fcnt)
write.csv(s,file="FinalRFile.csv",row.names=FALSE)
