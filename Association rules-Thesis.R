
d<-read.csv('I:/Thesis/Data/dm.csv',header = TRUE)
str(d)

#Demographic
d$Gender<-as.factor(d$Gender)
levels(d$Gender)=c("male","female")
d$marital.status<-as.factor(d$marital.status)
levels(d$marital.status)=c("single","married","widow","derelict")

d$Chest.Pain<-as.factor(d$Chest.Pain)
levels(d$Chest.Pain)=c("no","yes")
d$Syncope<-as.factor(d$Syncope)
levels(d$Syncope)=c("no","yes")
d$Severe.perspiration<-as.factor(d$Severe.perspiration)
levels(d$Severe.perspiration)=c("no","yes")
d$Killip.at.Admission<-as.factor(d$Killip.at.Admission)
levels(d$Killip.at.Admission)=c("Killip I","Killip II","Killip III","Killip IV")
d$Requires.VT...VF.intervention<-as.factor(d$Requires.VT...VF.intervention)
levels(d$Requires.VT...VF.intervention)=c("no","yes")
d$Bradycardia.and.asystole<-as.factor(d$Bradycardia.and.asystole)
levels(d$Bradycardia.and.asystole)=c("no","yes")
d$Arrest.outside.the.hospital<-as.factor(d$Arrest.outside.the.hospital)
levels(d$Arrest.outside.the.hospital)=c("no","yes")

#History
d$History.of.HF<-as.factor(d$History.of.HF)
levels(d$History.of.HF)=c("yes","dubious","no")
d$History.of.CABG<-as.factor(d$History.of.CABG)
levels(d$History.of.CABG)=c("yes","dubious","no")
d$History.of.PCI<-as.factor(d$History.of.PCI)
levels(d$History.of.PCI)=c("yes","dubious","no")
d$History.of.CVA<-as.factor(d$History.of.CVA)
levels(d$History.of.CVA)=c("yes","dubious","no")
d$Current.dialysis<-as.factor(d$Current.dialysis)
levels(d$Current.dialysis)=c("yes","dubious","no")
d$Diabetes<-as.factor(d$Diabetes)
levels(d$Diabetes)=c("yes","dubious","no")
d$Hypertension<-as.factor(d$Hypertension)
levels(d$Hypertension)=c("yes","dubious","no")
d$Hypercholesterolemia<-as.factor(d$Hypercholesterolemia)
levels(d$Hypercholesterolemia)=c("yes","dubious","no")
d$Recent.Smoker<-as.factor(d$Recent.Smoker)
levels(d$Recent.Smoker)=c("smoking","quit-smoking","never-smoking")
d$Opium<-as.factor(d$Opium)
levels(d$Opium)=c("no","yes")
d$Alcohol<-as.factor(d$Alcohol)
levels(d$Alcohol)=c("no","yes")
d$Father.history<-as.factor(d$Father.history)
levels(d$Father.history)=c("no","yes")
d$Mother.history<-as.factor(d$Mother.history)
levels(d$Mother.history)=c("no","yes")
d$Brother.history<-as.factor(d$Brother.history)
levels(d$Brother.history)=c("no","yes")
d$Sister.history<-as.factor(d$Sister.history)
levels(d$Sister.history)=c("no","yes")

d$BloodGroup<-as.factor(d$BloodGroup)
levels(d$BloodGroup)=c("A+","A-","B+","B-","AB+","AB-","O+","O-")
d$PCI<-as.factor(d$PCI)
levels(d$PCI)=c("not done","primary(PPCI)","rescue PCI","facilated PCI","urgent CABG")
d$Fibrinolytic.received<-as.factor(d$Fibrinolytic.received)
levels(d$Fibrinolytic.received)=c("no","yes")
d$Heart.rhythm<-as.factor(d$Heart.rhythm)
levels(d$Heart.rhythm)=c("sinus","non-sinus","pace")
d$MI.type<-as.factor(d$MI.type)
levels(d$MI.type)=c("STEMI","NSTEMI", "LBBB")

#Medicine
d$Calcium.Blocker<-as.factor(d$Calcium.Blocker)
levels(d$Calcium.Blocker)=c("no","yes")
d$Betblocker<-as.factor(d$Betblocker)
levels(d$Betblocker)=c("no","yes")
d$ACE.I.ARB<-as.factor(d$ACE.I.ARB)
levels(d$ACE.I.ARB)=c("no","yes")
d$Nitrate<-as.factor(d$Nitrate)
levels(d$Nitrate)=c("no","yes")
d$Digital<-as.factor(d$Digital)
levels(d$Digital)=c("no","yes")
d$Statin<-as.factor(d$Statin)
levels(d$Statin)=c("no","yes")
d$Insulin<-as.factor(d$Insulin)
levels(d$Insulin)=c("no","yes")
d$NSAID<-as.factor(d$NSAID)
levels(d$NSAID)=c("no","yes")
d$Aspirin<-as.factor(d$Aspirin)
levels(d$Aspirin)=c("no","yes")
d$Clopidogrel<-as.factor(d$Clopidogrel)
levels(d$Clopidogrel)=c("no","yes")
d$Aspirin1<-as.factor(d$Aspirin1)
levels(d$Aspirin1)=c("no","yes")
d$Clopidogrel1<-as.factor(d$Clopidogrel1)
levels(d$Clopidogrel1)=c("no","yes")
d$Heparin<-as.factor(d$Heparin)
levels(d$Heparin)=c("no","yes")
d$Clexan<-as.factor(d$Clexan)
levels(d$Clexan)=c("no","yes")
d$Integrilin<-as.factor(d$Integrilin)
levels(d$Integrilin)=c("no","yes")
d$blocker<-as.factor(d$blocker)
levels(d$blocker)=c("no","yes")
d$ACEI<-as.factor(d$ACEI)
levels(d$ACEI)=c("no","yes")
d$ARB<-as.factor(d$ARB)
levels(d$ARB)=c("no","yes")
d$CCB<-as.factor(d$CCB)
levels(d$CCB)=c("no","yes")
d$Nitrate1<-as.factor(d$Nitrate1)
levels(d$Nitrate1)=c("no","yes")
d$Statin1<-as.factor(d$Statin1)
levels(d$Statin1)=c("no","yes")
d$Diuretics<-as.factor(d$Diuretics)
levels(d$Diuretics)=c("no","yes")

d$Patient.status.at.discharge<-as.factor(d$Patient.status.at.discharge)
levels(d$Patient.status.at.discharge)=c("dead","alive")

str(d)

d1 <- d[,c(2,16:30,67)]

library(plyr)
library(arulesViz)
library(arules)

rules <- apriori(d1)
inspect(head(sort(rules,by="lift"),10))

#Rules with rhs containing "Survived" only
rules1 <- apriori(d1,parameter = list(minlen=2, supp=0.005, conf=0.8),
                  appearance = list(rhs=c("Patient.status.at.discharge=dead","Patient.status.at.discharge=alive"),
                                    default="lhs"),
                  control = list(verbose=F))
rules.sorted <- sort(rules1, by="lift")
inspect(rules.sorted)
inspect(head(rules.sorted,20))

# find redundant rules
subset.matrix <- is.subset(rules.sorted, rules.sorted)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
which(redundant)
# remove redundant rules
rules.pruned <- rules.sorted[!redundant]
inspect(rules.pruned)


#--------------------------------------------------Naive Bayez------------------------
library(caret)
indxTrain <- createDataPartition(y = d$Patient.status.at.discharge,p = 0.75,list = FALSE)
training <- d[indxTrain,]
testing <- d[-indxTrain,] 

#create objects x which holds the predictor variables and y which holds the response variables
x = training[,-67]
y = training$Patient.status.at.discharge

#Fit the model
library(e1071)
model = train(x,y,'nb')
model

#Plot Variable performance
X <- varImp(model)
plot(X)

#Model Evaluation
#Predict testing set
Predict <- predict(model,newdata = testing ) #Get the confusion matrix to see accuracy value and other parameter values 
confusionMatrix(Predict, testing$Patient.status.at.discharge )

newcases = read.csv('/Volumes/Leila LaCie/Stat Data Course/Naive Bayes/new cases.csv')
Predictnewcases <- predict(model,newdata = newcases )
Predictnewcases

#----------------------UnderSampling----------------------------------------------------------------
prop.table(table(d$Patient.status.at.discharge))

library(rpart)
treeimb <- rpart(Patient.status.at.discharge ~ ., data = hacide.train)
pred.treeimb <- predict(treeimb, newdata = hacide.test)

indxTrain <- createDataPartition(y = d$Patient.status.at.discharge,p = 0.75,list = FALSE)
training <- d[indxTrain,]
testing <- d[-indxTrain,] 

log.reg.imb <-  glm(Patient.status.at.discharge ~ ., data=training, family=binomial)
pred.log.reg.imb <- predict(log.reg.imb, newdata=testing,
                            type="response")
library(ROSE)
hacide.rose <- ROSE(Patient.status.at.discharge ~ ., data=training, seed=123)$data

table(hacide.rose$Patient.status.at.discharge)

# train logistic regression on balanced data
log.reg.bal <- glm(Patient.status.at.discharge ~ ., data=hacide.rose, family=binomial)

# use the trained model to predict test data
pred.log.reg.bal <- predict(log.reg.bal, newdata=testing,
                            type="response")

# check accuracy of the two learners by measuring auc
roc.curve(testing$Patient.status.at.discharge, pred.log.reg.imb)
roc.curve(testing$Patient.status.at.discharge, pred.log.reg.bal, add.roc=TRUE, col=2)

# determine bootstrap distribution of the AUC of logit models
# trained on ROSE balanced samples
# B has been reduced from 100 to 10 for time saving solely
boot.auc.bal <- ROSE.eval(Patient.status.at.discharge ~ ., data=training, learner= glm, 
                          method.assess = "BOOT", 
                          control.learner=list(family=binomial), 
                          trace=TRUE, B=10)

summary(boot.auc.bal)
