



import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import category_encoders 
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



# setting up streamlit dashboard and loading the data and the model 

st.set_page_config(layout='wide',page_title='Power Plants')
st.title('China Cleantech Exports - Energy Ember')

df=pd.read_csv('china_exports_cleaned.csv')

knn = joblib.load('knn.pkl')


# Dividing dashboard into multiple sections


# overview page

page=st.sidebar.radio('select page',['Overview','Analysis','Prediction'])


if page == 'Overview':
    st.dataframe(df)

    # explaining features meaning
    cols = {'Area':'Country or geographical area associated with the recorded data.',
            'Region	':'Geographical region to which the area belongs.',
            'Commodity category':'Clean technology category represented in the data.',
            'Amount (USD)':'Recorded monetary value for the commodity category in USD for the given month.',
            'Amount 12 month rolling sum (USD)':'Total monetary value over the preceding 12 months, including the current month.',
            'Year'	:'Calendar year of the observation.',
            'Month' :'month of the observation' }
                    
    for col, meaning in cols.items():
        with st.sidebar.expander(col):
            st.success(meaning)




    col1, col2, col3 , col4 = st.columns(4)

    total_countries = len(df['Area'].unique())
    avg_amount = round(df['Amount (USD)'].mean(),2)
    max_amount= round(df['Amount (USD)'].max(),2)




    with col1:
        with st.container(border=True):
            st.metric("Tota Countries", total_countries)


    with col2:
        with st.container(border=True):
            st.metric("Total Categories",6)


    
    with col3:
        with st.container(border=True):
            st.metric('Average Eports Amount',f'{avg_amount:,.0f} USD')

    
    with col4:
        with st.container(border=True):
            st.metric("Maximum Exports Aount", f"{max_amount:,.0f} USD")
    

        

    
       # analysis and visualization page

elif page == 'Analysis' :
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['Distributions','Exports by  Category',
                                             'Exports by Region','Time Series - Amount', 
                                             'Time Series - Seasonality by Category','Summary'])
    
    with tab1 :
        col6,col7= st.columns(2,gap = 'large')
       

        with col6 :

            fig_1 = px.histogram(df, x='Amount (USD)', title='Distribution of Exports Amount (USD)',nbins=5,marginal='rug',
                                color_discrete_sequence=['cornflowerblue'])
            fig_1.show()
            st.plotly_chart(fig_1,use_container_width=True)

        with col7 :
            
            fig_2 = px.histogram(df, x='Amount 12 month rolling sum (USD)', title='Distribution of Rolling Sum (USD)',nbins=5,marginal='rug',
                                color_discrete_sequence=['cornflowerblue'])
            fig_2.show()
            st.plotly_chart(fig_2,use_container_width=True)

    with tab2:
        category_totals = (df[df['Commodity category'] != 'All technologies'].groupby('Commodity category')['Amount (USD)'].sum().reset_index().sort_values('Amount (USD)',
                                                                                        ascending=False))


        bar_1= px.bar( category_totals, x='Commodity category',y='Amount (USD)',
            title='Exports by Commodity Category',color_discrete_sequence=['skyblue'])
        
        bar_1.show()


    with tab3:
        region_totals = (df[df['Region']!='Unknown'].groupby('Region')['Amount (USD)'].sum().reset_index().sort_values('Amount (USD)', ascending=False))


        bar_2= px.bar(region_totals,x='Region', y='Amount (USD)',
            title='Exports by Region',color_discrete_sequence=['deepskyblue'])
        
        bar_2.show()


    with tab4:
        
        mean = df.groupby('Year')['Amount (USD)'].mean().reset_index()
        line = px.line(mean,y='Amount (USD)',x='Year',title='Exports Trend Over Time',color_discrete_sequence=['dodgerblue'])
        line.show()


    with tab5:

        col7,col8,col9=st.columns(3,gap='large')
        col10,col11,col12=st.columns(3,gap='large')
        

        with col7:

            df_battery = df[df['Commodity category']=='Batteries'].groupby('Month')['Amount (USD)'].mean().reset_index()
        
            line_2 = px.line(  df_battery , x='Month',color_discrete_sequence=['deeppink'],
                y='Amount (USD)', markers=True, title='Monthly Seasonality - Batteries')
        
            line_2.show() 


        with col8:

            df_ev = df[df['Commodity category']=='EVs'].groupby('Month')['Amount (USD)'].mean().reset_index()
        
            line_3 = px.line( df_ev, x='Month',color_discrete_sequence=['deeppink'],
                y='Amount (USD)', markers=True, title='Monthly Seasonality - EVs')
        
            line_3.show() 

        with col9:
            df_grid = df[df['Commodity category']=='Grid'].groupby('Month')['Amount (USD)'].mean().reset_index()
        
            line_4= px.line( df_grid, x='Month',color_discrete_sequence=['deeppink'],
                y='Amount (USD)', markers=True, title='Monthly Seasonality - Grid')
        
            line_4.show() 


        with col10:
            
            df_heat = df[df['Commodity category']=='Heating and cooling systems'].groupby('Month')['Amount (USD)'].mean().reset_index()
        
            line_5= px.line( df_heat, x='Month',color_discrete_sequence=['deeppink'],
                y='Amount (USD)', markers=True, title='Monthly Seasonality - Heating and cooling systems')
        
            line_5.show() 

  
        



        with col11:
            
            df_solar = df[df['Commodity category']=='Solar PV'].groupby('Month')['Amount (USD)'].mean().reset_index()
        
            line_6= px.line( df_solar, x='Month',color_discrete_sequence=['deeppink'],
                y='Amount (USD)', markers=True, title='Monthly Seasonality - Solar PV')
        
            line_6.show() 



        
        with col12:
            
            df_wind = df[df['Commodity category']=='Wind'].groupby('Month')['Amount (USD)'].mean().reset_index()
        
            line_7= px.line( df_wind, x='Month',color_discrete_sequence=['deeppink'],
                y='Amount (USD)', markers=True, title='Monthly Seasonality - Wind')
        
            line_7.show() 

    with tab6:
        
        st.info('China exports value in cleantech has increased steadily since 2018.')
        st.info('There is a recurrent drop pattern in both February and October in all categories')
        st.info('Batteries are the top category in exports and Europe is the top region in investing in cleantech')
  



# prediction page

else:
    st.image("https://unsplash.com/photos/blue-solar-panel-boards-V4ZYJZJ3W4M")
    

  
    

    # Filters

    st.sidebar.header('Filters')

    country = st.sidebar.selectbox('country',df['Area'].unique())
    filtered_region = df[df['Area']==country]['Region'].unique()
    region = st.sidebar.selectbox('Region',filtered_region)

    category=st.sidebar.selectbox('Commodity category',df['Commodity category'].unique())

    amount = st.sidebar.slider('Amount (USD)',min_value=float(df['Amount (USD)'].min()),
                                                    max_value= float(df['Amount (USD)'].max()))


    year = st.number_input("Year",min_value=2018,max_value=2030,value=2027,step=1)


    month = st.selectbox('Month', list(range(1, 13)))




    
    # Prediction

    if st.button("Rolling Sum Forecasting"):
        df_prediction = pd.DataFrame(data= [[country,region,category,amount,year,month]],
                    columns=df.drop( 'Amount 12 month rolling sum (USD)',axis=1).columns)
        try:
            st.table(df_prediction)
            result = knn.predict(df_prediction)[0]
            prediction = knn.predict(df_prediction)
            st.success(f" Estimated Rolling Sum ': {prediction[0]:,.2f} USD")
        except Exception as e:
            st.error(f"Prediction Error: {e}")


