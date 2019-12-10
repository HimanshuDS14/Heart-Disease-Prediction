import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.metrics import accuracy_score , classification_report  ,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



data = pd.read_csv("Heart.csv")

print(data)


count = data["target"].value_counts()
print(count)

sns.heatmap(data.isnull() , yticklabels=False , cbar=False , cmap='viridis')
plt.show()

sns.countplot(x = data["target"] , data = data)
plt.show()

for i in data.columns:
    print(i)

target = np.array(data["target"])

age = np.array(data["age"])
sex = np.array(data["sex"])
cp = np.array(data["cp"])
trestbps = np.array(data["trestbps"])

chol = np.array(data["chol"])
fbs = np.array(data["fbs"])
restecg = np.array(data["restecg"])
thalach = np.array(data["thalach"])
exang = np.array(data["exang"])
oldpeak = np.array(data["oldpeak"])
slope = np.array(data["slope"])
ca = np.array(data["ca"])
thal = np.array(data["thal"])

print("age-->" , np.corrcoef(age , target).min())
print("sex-->" , np.corrcoef(sex , target).min())
print("cp-->" , np.corrcoef(cp , target).min())
print("trestbps-->" , np.corrcoef(trestbps , target).min())
print("chol-->" , np.corrcoef(chol , target).min())
print("fbs-->" , np.corrcoef(fbs , target).min())
print("restecg-->" , np.corrcoef(restecg , target).min())
print("thalach-->" , np.corrcoef(thalach , target).min())
print("exang-->" , np.corrcoef(exang , target).min())
print("oldpeak-->" , np.corrcoef(oldpeak , target).min())
print("slope-->" , np.corrcoef(slope , target).min())
print("ca-->" , np.corrcoef(ca , target).min())
print("thal-->" , np.corrcoef(thal , target).min())


x = data.drop(['target'] ,axis=1)
y = data["target"]

accuracy_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn ,x,y,cv=10)
    accuracy_rate.append(score.mean())

plt.scatter(range(1,40) , accuracy_rate , color = "green")
plt.plot(range(1,40) , accuracy_rate , color = "red" , linestyle = 'dashed')
plt.xlabel("K value")
plt.ylabel("accuracy rate")
plt.show()


train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=1/3 , random_state=0)

classifier = KNeighborsClassifier(n_neighbors=12)
classifier.fit(train_x , train_y)
pred = classifier.predict(test_x)

print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))
print(accuracy_score(test_y , pred))


rf  = RandomForestClassifier(n_estimators=50 , criterion='entropy')
rf.fit(train_x , train_y)
pred_y = rf.predict(test_x)

print(classification_report(test_y , pred_y))
print(confusion_matrix(test_y , pred_y))
print(accuracy_score(test_y , pred_y))


dtree = DecisionTreeClassifier(criterion="entropy")
dtree.fit(train_x , train_y)
pred_d = dtree.predict(test_x)

print(classification_report(test_y , pred_d))
print(confusion_matrix(test_y , pred_d))
print(accuracy_score(test_y , pred_d))


sample_data = np.array([[55,0,0,128,205,0,2,130,1,2,1,1,3]])
print(classifier.predict(sample_data))
print(rf.predict(sample_data))
print(dtree.predict(sample_data))
