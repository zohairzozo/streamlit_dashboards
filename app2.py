#Libraries 
import streamlit as st 
import plotly.express as px 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

# Make container 
header = st.container()
data_sets = st.container()
model_training = st.container()

with header: 
    st.title("Fertility Rate From 1960-2020 With Prediction For Future Years")
    st.text("We will work with Fertilty rate dataset collected from all over the world")
    
with data_sets:
    st.header("Fertility Dataset ")
  
    #import data 
    df = pd.read_csv('fertility_rate.csv')
    df = df.dropna()
    st.write(df.head(10))
    
    st.subheader("Graph that shows visualization for fertily rate per country per anum",)
    
    df1 = pd.melt(df, id_vars=["Country"], 
                  var_name="Years", value_name="Fertlity Rate")
    
    #Plotting 
    
    year_option = df1['Years'].unique().tolist()
    Year = st.sidebar.selectbox("You can see the Fertlity rate comparision of countries per year", year_option,0)

    fig = px.bar(df1, x="Country", y="Fertlity Rate", color="Country", hover_name="Country",
    animation_frame="Years", animation_group="Country" ,width=None, height=None)
    
    st.write(fig)
    
    country_option = df1['Country'].unique().tolist()
    Country = st.sidebar.selectbox("You can see how Fertlity rate varies in a country over time(years)", country_option,0)
    df1 = df1[df1['Country']== Country]
     
    fig1 = px.line(df1, x="Years", y="Fertlity Rate", hover_name="Country",
     width=None, height=None)
    
    st.write(fig1)
    
with model_training:
    
    X = df[["2017","2018","2019"]]
    y = df[["2020"]]
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model = model.fit(x_train, y_train)
    
    future, accuracy = st.columns(2)
    
    with future:
        a = st.number_input("Input a value of fertlity rate in 2017 for your country",min_value=None, max_value=None)
        b = st.number_input("Input a value of fertlity rate in 2018 for your country",min_value=None, max_value=None)
        c = st.number_input("Input a value of fertlity rate in 2019 for your country",min_value=None, max_value=None)
        
        predictions = model.predict([[a,b,c]])
        st.write("This is the prediction: ",predictions)
        
    with accuracy:
        accuracy = model.score(x_test,y_test)
        st.write('Score for Training data = ', accuracy)
        
        
        
        
    
    
    
    
    
