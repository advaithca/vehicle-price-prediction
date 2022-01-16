"""
This program shows the application of Linear Regression on a dataset of car details to predict the price of the 
car from the year of release and the kilometers driven. Dataset was downloaded from Kaggle.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import altair as alt
from sklearn.preprocessing import PolynomialFeatures

car_details = pd.read_csv(r"CAR_DETAILS_FROM_CAR_DEKHO_C.csv",na_values='?')
car_details.dropna()

st.write("""
# Vehicle-Price-Predict-inator !!!
""")

st.write("""
*******
""")

st.write("""
# Dataset Used
""")

car_details

train_details, test_details = train_test_split(car_details,train_size=0.8,shuffle=False)

g = alt.Chart(train_details).mark_point().encode(
    alt.X('year:Q',scale=alt.Scale(zero=False)),
    y = 'selling_price',
    tooltip = ['name','selling_price','year'],
)
reg = g.transform_regression('year','selling_price').mark_line(color="#FFAA00")

final_plot = (g + reg).interactive()

st.write("""
*******
""")

st.write("""
# A Plot of Kilometres driven vs Selling Price
""")
st.altair_chart(final_plot)

st.write("""
### It can be seen that Linear Model doesn't fit this graph well
""")

features = ["km_driven","year"]

x = np.array(train_details.loc[:,features])
y = np.array(train_details['selling_price'])

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

poly.fit(x_poly,y)

linmod = LinearRegression().fit(x_poly,y)

r_sq = linmod.score(x_poly,y)

st.write("""
Coefficient of determination : 
""",r_sq)

st.write("""
*******
""")

# new_features = ['km_driven','year']
# new_x = np.array(test_details.loc[:,new_features])

# new_x_poly = poly.fit_transform(new_x)

# y_pred = linmod.predict(new_x_poly)

# data = []
# j = 0

# for i in range(len(test_details)):
#     data.append([test_details.loc[i+3472,'name'],y_pred[j],test_details.loc[i+3472,'year']])
#     j += 1

# predictions = pd.DataFrame(data, columns=['Vehicle_Name','Predicted_price','Year'])

# st.write("""
# # Predicted prices

# ####
# """)
# predictions

# st.download_button(
#     label='Download as csv',
#     data=predictions.to_csv().encode('utf-8'),
#     file_name='carprice.csv',mime='text/csv'
# )

# st.write("""
# ######
# """)