# SPFX
3rd Year Project
A comparison of two ensemble algorithms on a forecast of the Welsh population:

        ◦ Trained on subnational populations figures of Australia, The Netherlands, Sweden and Denmark ranging from 1942-2024
            ▪ Used imputation to handle missing values
        ◦ Evaluated using RMSE, MAE, MAPE, R2 and Computation Time.
        ◦ Created a stacking ensemble in Scikit-learn (StackingRegressor) using the K-Nearest Neighbour, Multi-Layer Perceptron as candidate models and a linear regression model as the meta-learner, which was then compared against a Random Forest. The stacking ensemble was the most performant with an RMSE of 61292 and an R2 of 0.991.
        ◦ The data and models were integrated into a web app developed with Streamlit. The app also includes extra features that allow simple configuration of the models

Tools Used: Python (Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib) 
