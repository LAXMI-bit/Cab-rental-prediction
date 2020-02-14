rm(list=ls(all=T))
setwd("/Users/ravindranlakshmanapillai/Desktop/Cab fare prediction")
getwd()


#LOAD LIBRARIES ##

x=c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
    "DataCombine", "doSNOW", "inTrees", "rpart.plot", "rpart",'MASS','xgboost','stats')

install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)




## LOAD THE GIVEN TRAIN DATA ###

train = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test=read.csv('test.csv')

summary(train)
summary(test)
#Check data shape ##
str(train)

# Note: fare_amount is factor and remaining are numeric ####
#convert fare_amount to Numeric type and Passenger to Integer type##

train$fare_amount=as.numeric(as.character(train$fare_amount), error = "coercion")
str(train)
train$passenger_count=as.integer(train$passenger_count)
str(train)

### Eliminate cells with same pickup and dropoff location

train=subset(train, !(train$pickup_longitude==train$dropoff_longitude & train$pickup_latitude==train$dropoff_latitude))

##replace "0's" with NA

train[train==0]= NA

### Removing values which are not within desired range(outlier) depending upon basic understanding of dataset.

# 1.Fare amount has a negative value, which doesn't make sense. A price amount cannot be -ve and also cannot be 0. So we will remove these fields.
train[which(train$fare_amount < 1 ),]
nrow(train[which(train$fare_amount < 1 ),])
train = train[-which(train$fare_amount < 1 ),]

train[which(train$fare_amount < 1 ),]

#2.Passenger_count variable
for (i in seq(6,15,by=1)){
  print(paste('passenger_count above ' ,i,nrow(train[which(train$passenger_count > i ),])))
}

# so 18 observations of passenger_count is consistenly above from 6,7,8,9,10 passenger_counts, let's check them.
train[which(train$passenger_count > 6 ),]


# we need to see if there are any passenger_count==0
train[which(train$passenger_count <1 ),]
nrow(train[which(train$passenger_count <1 ),])

# We will remove 18 observation which are above 6 value because a cab cannot hold these number of passengers.
train = train[-which(train$passenger_count > 6),]
nrow(train[which(train$passenger_count >6 ),])

##3.Latitudes range from -90 to 90.Longitudes range from -180 to 180.Removing which does not satisfy these ranges

print(paste('pickup_longitude above 180=',nrow(train[which(train$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train[which(train$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(train[which(train$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(train[which(train$pickup_latitude < -90 ),])))
print(paste('dropoff_longitude above 180=',nrow(train[which(train$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(train[which(train$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(train[which(train$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train[which(train$dropoff_latitude > 90 ),])))

# There's only one outlier which is in variable pickup_latitude.So we will remove it with nan.

# Also we will see if there are any values equal to 0.
nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])

train = train[-which(train$pickup_latitude > 90),]


df=train
train=df

###Missing Value Analysis ##############


#########function to calculate missing values ###################
missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
missing_val

unique(train$passenger_count)
unique(test$passenger_count)

train[,'passenger_count'] = factor(train[,'passenger_count'], labels=(1:6))
test[,'passenger_count'] = factor(test[,'passenger_count'], labels=(1:6))


# 1.For Passenger_count:
# Actual value = 1
# Mode = 1
# KNN = 1
train$passenger_count[1000]
train$passenger_count[1000] = NA
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

sapply(train, sd, na.rm = TRUE)

train$fare_amount[1000]
train$fare_amount[1000]= NA



###########As PAssenger_count is a categorical Variable , so we will use mode for Imputation######### 

##########calculate mode - create function ###########
mode= function(data){
  uniq=unique(data)
  as.numeric(as.character(uniq[which.max(tabulate(match(data,uniq)))]))
  #print(mode_d)
}
mode(train$passenger_count)
getmode(train$passenger_count)


mean(train$fare_amount, na.rm = T)

median(train$fare_amount, na.rm = T)

train = knnImputation(train, k = 3)
train$fare_amount[1000]
train$passenger_count[1000]
sapply(train, sd, na.rm = TRUE)

sum(is.na(train))
str(train)
summary(train)





#####################OUTLIER ANALYSIS############################################

############outliers in fare_amount
#Remove negative values from 'fare_amount'
train$fare_amount=ifelse(train$fare_amount<0, NA, train$fare_amount)
train$fare_amount=ifelse(train$fare_amount>454,NA, train$fare_amount)

# We Will do Outlier Analysis only on Fare_amount and we will do outlier analysis after feature engineering laitudes and longitudes.
# Boxplot for fare_amount
pl1 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# Replace all outliers with NA and impute
vals = train[,"fare_amount"] %in% boxplot.stats(train[,"fare_amount"])$out
train[which(vals),"fare_amount"] = NA

#lets check the NA's
sum(is.na(train$fare_amount))

#Imputing with KNN
train = knnImputation(train,k=3)

# lets check the missing values
sum(is.na(train$fare_amount))
str(train)

df2=train
train=df2

#########################FEATURE SCALING/ENGINEERING######################

# 1.Feature Engineering for timestamp variable
# we will derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time

train$pickup_date = as.Date(as.character(train$pickup_datetime))
train$pickup_weekday = as.factor(format(train$pickup_date,"%u"))# Monday = 1
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train$pickup_hour = as.factor(format(pickup_time,"%H"))

#Add same features to test set
test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$pickup_hour = as.factor(format(pickup_time,"%H"))

sum(is.na(train))

train = subset(train,select = -c(pickup_datetime,pickup_date))
test = subset(test,select = -c(pickup_datetime,pickup_date))

summary(train)
summary(test)

#create new variable
library(geosphere)
train$dist= distHaversine(cbind(train$pickup_longitude, train$pickup_latitude), cbind(train$dropoff_longitude,train$dropoff_latitude))
#the output is in metres, Change it to kms
train$dist=as.numeric(train$dist)/1000
df=train
train=df

test$dist=distHaversine(cbind(test$pickup_longitude, test$pickup_latitude), cbind(test$dropoff_longitude,test$dropoff_latitude))
test$dist=as.numeric(test$dist)/1000
                        
###########################################CORRELATION AMALYSIS ####################################

library(corrgram)
corrgram(train[,-6], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#####correlation between the numeric variables

num_cor=round(cor(train[,-6]), 3)


#Eliminate the pickup and dropoff locations if same (if any)

train=subset(train, !(train$pickup_longitude==train$dropoff_longitude & train$pickup_latitude==train$dropoff_latitude))

#######remove unnecessary variables

rm(missing_val,pickup_time,pl1,cnames,i,val)

########################## MODEL DEVELOPMENT ###################################################3

#create sampling and divide data into train and test

set.seed(123)
train_index = sample(1:nrow(train), 0.8 * nrow(train))

train = train[train_index,]
test = train[-train_index,]

########### Define Mape - The error matrix to calculate the error and accuracy ################

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y*100))
}

############################################Decision Tree#####################################

library(rpart)
fit = rpart(fare_amount ~. , data = train, method = "anova")

summary(fit)
predictions_DT = predict(fit, test[,-1])
regr.eval(test[,1],predictions_DT)

MAPE(test[,1], predictions_DT)

write.csv(predictions_DT, "Prediction_DT.csv", row.names = F)

#Error 20.63
#Accuracy 79.37


########################################Random Forest###############################################

library(randomForest)
RF_model = randomForest(fare_amount ~.  , train, importance = TRUE, ntree=200)
RF_Predictions = predict(RF_model, test[,-1])
regr.eval(test[,1],RF_Predictions)
MAPE(test[,1], RF_Predictions)
importance(RF_model, type = 1)

#error 9.911 for n=100
#accuracy = 90.089




######################################Linear Regression###########################################################

lm_model = lm(fare_amount ~. , data = train)
summary(lm_model)

predictions_LR = predict(lm_model, test[,-1])
MAPE(test[,1], predictions_LR)

#error 26.22016
#Accuracy 73.78

#####################################KNN Implementation############################################################


KNN_Predictions = knn(train[, 2:7], test[, 2:7], train$fare_amount, k = 1)

#convert the values into numeric
KNN_Predictions=as.numeric(as.character((KNN_Predictions)))

#Calculate MAPE
MAPE(test[,1], KNN_Predictions)*100

#error 28.75
#Accuracy = 71.25



##############Model Selection and Final Tuning##########################

#Random Forest with using mtry = 2 that is fixing only two variables to split at each tree node 

RF_model = randomForest(fare_amount ~.  , train, importance = TRUE, ntree=200, mtry=2)
RF_Predictions = predict(RF_model, test[,-1])
MAPE(test[,1], RF_Predictions)
importance(RF_model, type = 1)

#error 10.71 for n=200
#Accuracy 89.29



###################################Predict VAlues in Test Data###################

pred_data=read.csv("test.csv", header= T)[,-1]

#########create distance variable
pred_data=subset(pred_data, !(pred_data$pickup_longitude==pred_data$dropoff_longitude & pred_data$pickup_latitude==pred_data$dropoff_latitude))
pred_data[pred_data==0]= NA

# COnnvert Data into proper data types

str(pred_data)
pred_data$passenger_count=as.factor(pred_data$passenger_count)

#calculate distance

pred_data$dist= distHaversine(cbind(pred_data$pickup_longitude, pred_data$pickup_latitude), cbind(pred_data$dropoff_longitude,pred_data$dropoff_latitude))

#the output is in metres, Change it to kms

pred_data$dist=as.numeric(pred_data$dist)/1000

# Create the target variable
pred_data$fare_amount=0
pred_data=pred_data[,c(1,2,3,4,5,6,7)]

#Random Forest
RF_model = randomForest(fare_amount ~.  , test, importance = TRUE)
pred_data$fare_amount = predict(RF_model, pred_data)

write.csv(pred_data, "Predicted_Data.csv", row.names = F)
