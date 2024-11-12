import pandas as pd
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
from matplotlib import pyplot as plt
import joblib

file=pd.read_csv (r'C:\Users\Srikkanth\Desktop\Files\LoanDefaultData.csv')

le=preprocessing.LabelEncoder() #Encoding the string values into labels 
F = file.apply(le.fit_transform)

print(F.head())

#Univariate analysis
plt.figure(1);
plt.subplot(221);
file['loan_duration'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='loan_duration');


plt.subplot(222);
file['grade'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='grade');

plt.subplot(223);
file['loan_amount'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='loan_amount');


plt.subplot(224);
sn.distplot(file['annual_pay'])


#Bivariate analysis and heatmap

IT=pd.crosstab(file['income_type'],file['is_default'])
IT.div(IT.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

GD=pd.crosstab(file['grade'],file['is_default'])
GD.div(GD.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

OT=pd.crosstab(file['own_type'],file['is_default'])
OT.div(OT.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))


AT=pd.crosstab(file['app_type'],file['is_default'])
AT.div(AT.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

IP=pd.crosstab(file['interest_payments'],file['is_default'])
IP.div(IP.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))


bins = [0.0,2.5,5.0,7.5,10.0];  group=['low', 'avg', 'high', 'vhigh'];
file['Empdur']=pd.cut(file['emp_duration'],bins, labels = group);
Empdur=pd.crosstab(file['Empdur'],file['is_default']);
Empdur.div(Empdur.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('Emp_duration');
p=plt.ylabel('Ptage')

bins = [0,50000,100000,150000,200000,250000];  group=['vlow', 'low', 'avg', 'high', 'vhigh'];
file['Annualpay']=pd.cut(file['annual_pay'],bins, labels = group);
Annualpay=pd.crosstab(file['Annualpay'],file['is_default']);
Annualpay.div(Annualpay.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('AnnualPay');
p=plt.ylabel('Ptage')

bins = [0,3000,6000,9000,15000,20000,25000];  group=['vlow', 'low', 'avg', 'high', 'vhigh','vvhigh'];
file['LoanAmount']=pd.cut(file['loan_amount'],bins, labels = group);
LoanAmount=pd.crosstab(file['LoanAmount'],file['is_default']);
LoanAmount.div(LoanAmount.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('LoanAmount');
p=plt.ylabel('Ptage')

bins = [0.0,2.5,5.0,7.0,10.0,15.0];  group=['vlow', 'low', 'avg', 'high', 'vhigh'];
file['TotalPay']=pd.cut(file['total_pymnt'],bins, labels = group);
TotalPay=pd.crosstab(file['TotalPay'],file['is_default']);
TotalPay.div(TotalPay.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('TotalPayment');
p=plt.ylabel('Ptage')

bins = [5.0,8.0,10.0,12.0,15.0,20.0];  group=['vlow', 'low', 'avg', 'high', 'vhigh'];
file['InterestRate']=pd.cut(file['interest_rate'],bins, labels = group);
InterestRate=pd.crosstab(file['InterestRate'],file['is_default']);
InterestRate.div(InterestRate.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('InterestRate');
p=plt.ylabel('Ptage')



matrix=file.corr();


f,ax=plt.subplots(figsize=(9,6))
sn.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


y=F.is_default
X=F.drop(['is_default','state', 'year','date_issued', 'date_final','loan_purpose','cust_id',],axis=1)

print(F.columns)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)          #Test and train split


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X)



model=LogisticRegression()                                                        #Applying logistic regression
model.fit(X_train, y_train)
LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=1, solver='lbfgs', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1, l1_ratio=None)
pred_test=model.predict(X_test)

print(accuracy_score(y_test,pred_test))
joblib.dump(model, 'LR.pkl')