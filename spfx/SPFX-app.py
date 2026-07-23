import streamlit
import pandas
import seaborn
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from skforecast.direct import ForecasterDirect

class backend:

    @streamlit.cache_data
    def wales_data():
        df = pandas.read_csv('WalesData/WalesPops-Transposed.csv')
        df['Date'] = pandas.to_datetime(df['Date'], format='%Y')
        df.set_index('Date', inplace=True)
        df = df.asfreq('YS')

        return df

    @streamlit.cache_data
    def training_data():
        return pandas.read_csv('aggdata/aggpops (copy 1).csv')
    
    
    @streamlit.cache_resource
    def ensembles():
        models = {
            'stackensemble': joblib.load('stackensemble'),
            'randomforest': joblib.load('randforest')
        }

        return models

streamlit.title('Subnational Population Forecast of Wales SPFX')

tab1, tab2, tab3 = streamlit.tabs(['Home', 'Partial Forecast', 'Full Forecast'])


with tab1:
    default_data = backend.wales_data()
	
    select_data = streamlit.segmented_control(
        label='Select Data', 
        options=['Default', 'Custom'], 
        selection_mode='single', 
        default='Default'
    )
    
    final_data_choice = ...
    
    if select_data == 'Default':
        final_data_choice = backend.wales_data()
    else:
        custom_data = streamlit.file_uploader(
        label='Upload File',
        type=['csv', 'xlsx']
        )
        
        final_data_choice = custom_data
        
    streamlit.subheader('Testing Data', divider=True)
    streamlit.dataframe(final_data_choice)
    
    column = streamlit.selectbox(label='Pick local authority', options=final_data_choice.columns)

if column:
    streamlit.bar_chart(final_data_choice[f'{column}'])

    ############ Training Data 
    streamlit.subheader('Training Data', divider=True)

    impute = streamlit.radio(
            label='Choose training data with or without imputation',
            options=['With Imputation', 'Without Imputation'],
            index=1
        )

    if impute == 'With Imputation':
        imputed_df = SimpleImputer().set_output(transform='pandas').fit_transform(backend.training_data().drop('Name', axis=1))
        imputed_df.insert(loc=0, column='Name', value=backend.training_data()['Name'])
        streamlit.dataframe(imputed_df)
    else:
        streamlit.dataframe(backend.training_data())

    traindf_column = streamlit.selectbox(label='Pick area', options=backend.training_data().columns, index=1)

    streamlit.bar_chart(backend.training_data()[f'{traindf_column}'])

with tab2:
    set_lags = streamlit.number_input(label='set number of lags', min_value=1, max_value=20, value=1)
    set_steps = streamlit.number_input(label='set number of predictions', min_value=1, max_value=20, value=10)

    counties = streamlit.selectbox(label='pick a local authority', options=final_data_choice.columns)

    if set_lags and set_steps:
        models = backend.ensembles()
        stackensemble = ForecasterDirect(
            estimator=models['stackensemble'],
            lags=set_lags,
            steps=set_steps,
            differentiation=1
        )
        
        randomforest = ForecasterDirect(
            estimator=models['randomforest'],
            lags=set_lags,
            steps=set_steps,
            differentiation=1
        )

    models_list = {
        'Stacking Ensemble': stackensemble,
        'Random Forest': randomforest
    }

    chosen_model = streamlit.selectbox(label='choose model', options=models_list.keys())
    
    if streamlit.button(label='fit and predict') and counties and chosen_model:
        models_list[chosen_model].fit(final_data_choice[counties])
        new_forecast = models_list[chosen_model].predict()
        
        fig, ax = plt.subplots()
        ax.plot(final_data_choice[f'{counties}'])
        ax.plot(new_forecast)
        
        streamlit.pyplot(fig)

with tab3:

    @streamlit.cache_data
    def full_forecast():
        forecasts = {
            'Stacking Ensemble': {},
            'Random Forest': {}
        }
        
        for model_name, model in models_list.items():
            for column in final_data_choice.columns:
                model.fit(final_data_choice[column])
                forecasts[model_name][column] = model.predict()
        
        return forecasts
    
    full_forecasts = full_forecast()

    streamlit.subheader('Stacking Ensemble')
    streamlit.dataframe(pandas.DataFrame(full_forecasts['Stacking Ensemble']))

    streamlit.subheader('Random Forest')
    streamlit.dataframe(pandas.DataFrame(full_forecasts['Random Forest']))


    if counties:
        fig, ax = plt.subplots(1,2, figsize=(15,12))
        ff_selectbox = streamlit.selectbox(label='Select Region', options=final_data_choice.columns)

        if ff_selectbox:
            ax[0].plot(full_forecasts['Stacking Ensemble'][ff_selectbox])
            ax[0].set_title('Stacking Ensemble')
            
            ax[1].plot(full_forecasts['Random Forest'][ff_selectbox])
            ax[1].set_title('Random Forest')
            
            streamlit.pyplot(fig)
            
