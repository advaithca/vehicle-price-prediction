import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv(r"D:\Documents\Python\CAR_DETAILS_FROM_CAR_DEKHO_C.csv")
test_details = pd.read_csv(r"D:\Documents\Python\car_data.csv")
new_data = []
with open(r"D:\Documents\Python\CAR_DETAILS_FROM_CAR_DEKHO_C.csv",'r') as f:
    creader = csv.reader(f)
    dreader = csv.DictReader(f)

    header = dreader.fieldnames
    for row in creader:
        if row[4] == "Diesel":
            row[4] = 1
        elif row[4] == "Petrol":
            row[4] = 2
        elif row[4] == "CNG":
            row[4] = 3
        else:
            row[4] = 4
        

        if row[5] == "Individual":
            row[5] = 0
        elif row[5] == "Dealer":
            row[5] = 1
        else :
            row[5] = 2
        
        if row[6] == "Manual":
            row[6] = 1
        elif row[6] == "Automatic":
            row[6] = 2
        else :
            row[6] = 0
        
        if row[7] == "First Owner":
            row[7] = 1
        elif row[7] == "Second Owner":
            row[7] = 2
        elif row[7] == "Third Owner":
            row[7] = 3
        else :
            row[7] = 4
        
        new_data.append(row)

t_data = pd.DataFrame(data=new_data,columns=header)

train_data = t_data.loc[:,header[1:]]

xdata = train_data

scaler = StandardScaler()

scaler.fit(xdata)

train = scaler.transform(xdata)

pca = PCA(0.95)

pca.fit(train_data)

train = pca.transform(train)



features = ["year", "km_driven", "fuel", "transmission"]

X = train
y = train_data.loc[:,'selling_price']

linmod = LinearRegression().fit(X,y)


# r_sq = linmod.score(X,y)

# print(f"Coeffn of prediction : {r_sq}")

ntest_data = []
with open(r"D:\Documents\Python\car_data.csv",'r') as f:
    creader = csv.reader(f)
    dreader = csv.DictReader(f)

    headers = dreader.fieldnames
    for row in creader:
        if row[5] == "Diesel":
            row[5] = 1
        elif row[5] == "Petrol":
            row[5] = 2
        elif row[5] == "CNG":
            row[5] = 3
        else:
            row[5] = 4

        if row[6] == "Individual":
            row[6] = 0
        elif row[6] == "Dealer":
            row[6] = 1
        else :
            row[6] = 3

        if row[7] == "Manual":
            row[7] = 1
        elif row[7] == "Automatic":
            row[7] = 2
        else :
            row[7] = 0
        
        ntest_data.append(row)

test_data = pd.DataFrame(data=ntest_data,columns=headers)
nheader = [headers[1],headers[2],headers[4],headers[5],headers[6],headers[7],headers[8]]
dt = test_data.loc[:,nheader]

xdata = dt

scaler = StandardScaler()

scaler.fit(xdata)

train = scaler.transform(xdata)

pca = PCA(0.95)

pca.fit(train_data)

train = pca.transform(train)


y_pred = linmod.predict(train)

data = []
j = 0

for i in test_data['Car_Name']:
    data.append([i,y_pred[j]])
    j += 1

print(linmod.score(train,y))
predictions = pd.DataFrame(data, columns=['Vehicle_Name','Predicted_price'])

# print(predictions)

pred = predictions.to_markdown()

print(pred)