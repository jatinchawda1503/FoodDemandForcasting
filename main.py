from turtle import color, width
from unicodedata import category
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


def week_order_line(df):
    dfg = df.groupby(["week"])["num_orders"].sum()
    fig = px.line(x=dfg.index, y=dfg)
    fig.update_layout(title='Pattern of Orders',
                   xaxis_title='Week',
                   yaxis_title='Number of Order per Week')
    return fig
st.plotly_chart(week_order_line(data), use_container_width=True)

def checkout_orders_hist(df):
    fig = px.histogram(df, x="checkout_price", y="num_orders", marginal="rug",
                    hover_data=df.columns)
    fig.update_layout(title='Checkout Price Vs Number of Orders',
                   xaxis_title='Checkout Price',
                   yaxis_title='Number of Orders')
    return fig
st.plotly_chart(checkout_orders_hist(data), use_container_width=True)



def order_with_cat_weekly_line(df):

    option = df['category'].unique().tolist()

    options = st.multiselect(
        'Select Category',
        option,
        option
    )

    dfs = {cat: df[df["category"] == cat] for cat in options}

    fig = go.Figure()
    for cat, orders in dfs.items():
        dfg = orders.groupby(["week"])["num_orders"].sum()
        fig = fig.add_trace(go.Scatter(x=dfg.index, y=dfg, name=cat))
        fig.update_layout(title='Weekly Number of Orders based on category',
                   xaxis_title='Week',
                   yaxis_title='Number of Orders')

    return fig
        
st.plotly_chart(order_with_cat_weekly_line(data), use_container_width=True)


def cat_center_hist(df):
    fig = px.histogram(df, x="category", y="num_orders",
                color='center_type', barmode='group')
    return fig
st.plotly_chart(cat_center_hist(data), use_container_width=True)



##Dist Plot 
def base_checkout(df):
    hist_data = [df['base_price'],df['checkout_price']]
    group_labels = ['base_price','checkout_price']
    fig = ff.create_distplot(hist_data, group_labels,bin_size=[1, 1])
    return fig
st.plotly_chart(base_checkout(data), use_container_width=True)


#Base price and checkout price are pretty much similar for each command by cuisine, nevertheless there are some instances where base price exceeds the checkout price. The most noticable one is within the Continental cuisine.

# def single_base_price(df):
#     return df["base_price"] / df["num_orders"]


# def single_checkout_price(df):
#     return df["checkout_price"] / df["num_orders"]


# data["single_base_price"] = data.apply(lambda x: single_base_price(x), axis=1)
# data["single_checkout_price"] = data.apply(lambda x: single_checkout_price(x), axis=1)



# def base_with_cat_weekly_line(df):
    
#     option = df['category'].unique().tolist()

#     option_cat = st.multiselect(
#         'Select Category',
#         option,
#         option
#     )

#     dfs = {cat: df[df["category"] == cat] for cat in option_cat}

#     fig = go.Figure()
#     for cat, orders in dfs.items():
#         dfg = orders.groupby(["week"])["single_base_price"].sum()
#         fig = fig.add_trace(go.Scatter(x=dfg.index, y=dfg, name=cat))
#         fig.update_layout(title='Weekly Number of Orders based on category',
#                    xaxis_title='Week',
#                    yaxis_title='Number of Orders')

#     return fig
        
# st.plotly_chart(base_with_cat_weekly_line(data), use_container_width=True)