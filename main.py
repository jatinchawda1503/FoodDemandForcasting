from turtle import color, width
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Welcome to FDA")


def read_data():
    train = pd.read_csv('data/train.csv')
    fulfilment_center_info = pd.read_csv('data/fulfilment_center_info.csv')
    meal = pd.read_csv('data/meal_info.csv')
    data_train_center_merge = pd.merge(train, fulfilment_center_info, on='center_id')
    data_merged = pd.merge(data_train_center_merge, meal, on='meal_id')
    return data_merged


data = read_data()
st.write(data)

st.subheader('Weekly Demand Data')


def cat_order_bar(df):
    dfg = df.groupby(["category"])["num_orders"].sum().sort_values(ascending=False)
    fig = px.bar(x=dfg.index, y=dfg, color=dfg.index,title="Orders by Category in Millions",text_auto='.2s')
    return fig


st.plotly_chart(cat_order_bar(data), use_container_width=True)



def cuisine_order_pie(df):
    fig = px.pie(df,'cuisine','num_orders',title='Percentage of Orders irresprctive of cuisine')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

st.plotly_chart(cuisine_order_pie(data), use_container_width=True)

def order_center_pie(df):
    fig = px.pie(df,'center_type','num_orders',title='Percentage of Orders irresprctive of Center')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

st.plotly_chart(order_center_pie(data), use_container_width=True)

def box_relation(df):
    cols = df[['week','checkout_price','base_price','num_orders']]

    plot_rows=2
    plot_cols=2
    fig = make_subplots(rows=2, cols=2)

    # add traces
    x = 0
    for i in range(1, plot_rows + 1):
        for j in range(1, plot_cols + 1):
            fig.add_trace(go.Box(name= cols.columns[x],
                                y = cols[cols.columns[x]].values
                                ),
                        row=i,
                        col=j)

            x=x+1
    fig.update_layout(width=700,height=700,title_text="Relationship of Week, Checkout Price, Base Price and Number of Orders")
    return fig

st.plotly_chart(box_relation(data), use_container_width=True)

