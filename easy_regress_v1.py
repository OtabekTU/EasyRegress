# Importing all the libraries 
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import itertools
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsaplots 
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.diagnostic import het_arch
import datetime


# Defining a function for the Plotting exercise 
@st.cache_resource
def plotting_exercise(regressand, label_name, dates_checkbox = False, dates = None):
    fig3, ax = plt.subplots(figsize=(8, 6))

    if dates_checkbox:
        dates = pd.to_datetime(dates, dayfirst=True)
        last_index = len(regressand)

        # Alighning the index of the regressand and the index
        dates_copy = dates[:last_index]
        index = dates_copy

        ax.plot(index, regressand) # at this moment, the only thing that is different is the name of the variable that is being plotted
        ax.set_title(f"Plot of {label_name}")
        ax.set_ylabel(label_name)
        #ax.set_xlabel("Dates")
        ax.set_xticks(index) # common part
        n_to_plot = len(index) / 10 # common part
        n_to_plot = int(round(n_to_plot)) # common part
        every_other_date = index[::n_to_plot] # common part
        ax.set_xticks(every_other_date) # common part
        ax.tick_params(axis='x', rotation=45) # common part                    
                        
    else:

        index = np.arange(len(regressand))
        ax.plot(index, regressand)
        ax.set_title(label_name)
        ax.set_ylabel(label_name)

        #ax.set_xlabel("Observations")
        ax.set_xticks(index)
        n_to_plot = len(index) / 10
        n_to_plot = int(round(n_to_plot))
        every_other_date = index[::n_to_plot]
        ax.set_xticks(every_other_date)
        ax.tick_params(axis='x', rotation=45)  

    plt.tight_layout()
    st.pyplot(fig3)

# Defining a function to draw ACF and PACF
@st.cache_resource
def acf_pacf(series1, series2):
    # Plot ACF on the first subplot (axes[0])
    fig13, axes = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(series1, ax=axes[0])
    axes[0].set_title('ACF')

    # Plot PACF on the second subplot (axes[1])
    plot_pacf(series2, ax=axes[1])
    axes[1].set_title('PACF')

    # Display the figure in Streamlit
    plt.tight_layout()
    st.pyplot(fig13)

# Defining a function for the manual ARIMA model
@st.cache_resource
def conduct_manual_arima(in_sample_arima_y, one, two, three):
    model = ARIMA(in_sample_arima_y, order = (int(one), int(two), int(three)))
    return model

# Defining a function to run an auto ARIMA model
@st.cache_resource
def conduct_auto_arima(in_sample_arima_y):
    auto_arima = pm.auto_arima(in_sample_arima_y, stepwise = False, seasonal = False)
    return auto_arima

# Defining a function toarima_ draw the KDE with residuals plot
@st.cache_resource
def resid_kde(_model_fit):    
    residuals = _model_fit.resid[1:]
    fig, axes = plt.subplots(1,2, figsize=(8, 6))
    residuals.plot(title = "Residuals", ax = axes[0])
    residuals.plot(title = "Density", kind = "kde", ax = axes[1])
    return fig, residuals

# Defining a color coding function 
@st.cache_resource
def color_code(row):
    # Find the minimum and maximum values in the row
    min_value = row.min()
    max_value = row.max()

    # Initialize a list to store the styles
    styles = []

    # Loop through each element in the row
    for value in row:
        if value == max_value:
            styles.append('background-color: red')  # Highlight max value
        elif value == min_value:
            styles.append('background-color: green')  # Highlight min value
        else:
            styles.append('background-color: white')  # No highlight for other values

    return styles

# Defining all the functions for post estiamtion diagnostics
@st.cache_resource
def gauss_markov(fitted_values, residuals):
    st.markdown("<h2 style = 'text-align: center;'>Gauss Markov assumptions assessments</h2>", unsafe_allow_html = True)
    fig3, axes3 = plt.subplots(3, 2, figsize = (12, 4 * 3)) 

    # Adding heteroscedasticity plot
    # Plotting the residuals against the zero line
    axes3[0, 0].scatter(fitted_values, residuals, marker = '.')
    axes3[0, 0].axhline(0, color = 'red', linestyle = '--', lw = 2)
    axes3[0, 0].set_title("Scatter plot of residuals and fitted values (Het)")
    axes3[0, 0].set_xlabel("Fitted values")
    axes3[0, 0].set_ylabel("Residuals")

    # Adding serial correlation plots
    axes3[0, 1].plot(residuals, marker = '.', linestyle = 'none')
    axes3[0, 1].axhline(0, color = 'red', linestyle = '--', lw = 2)
    axes3[0, 1].set_xlabel('Observation')
    axes3[0, 1].set_ylabel('Residuals')
    axes3[0, 1].set_title('Residuals from the Regression Model (Serial Cor)')

    tsaplots.plot_acf(residuals, lags=20, ax = axes3[1, 0])
    axes3[1, 0].set_xlabel('Lags')
    axes3[1, 0].set_ylabel('Autocorrelation')
    axes3[1, 0].set_title('ACF of Residuals (Serial Cor)')

    # Step 4: Plot Partial Autocorrelation Function (PACF)
    tsaplots.plot_pacf(residuals, lags=20, ax = axes3[1, 1])
    axes3[1, 1].set_xlabel('Lags')
    axes3[1, 1].set_ylabel('Partial Autocorrelation')
    axes3[1, 1].set_title('PACF of Residuals (Serial Cor)')

    # Visualising serial correlation of errors
    # Histogram with KDE
    sns.histplot(residuals, kde=True, stat="density", linewidth=0, ax = axes3[2, 0])
    # Plot the PDF
    x_norm = np.linspace(0 - 4*1, 0 + 4*1, 1000) # where 0 is a mean and 1 is a st.dev of normal distribution
    y_norm = stats.norm.pdf(x_norm, 0, 1) # x, mean, std_dev
    axes3[2, 0].plot(x_norm, y_norm, label=f'Normal Distribution\nMean = {0}, Std Dev = {1}', color = 'black')
    axes3[2, 0].set_title("Histogram of Residuals with KDE (Norm of errs)")
    axes3[2, 0].set_xlabel('Residuals')
    axes3[2, 0].set_ylabel('Density')
    axes3[2, 0].legend()

    # Q-Q Plot
    sm.qqplot(residuals, line='45', fit=True, ax = axes3[2, 1])
    axes3[2, 1].set_title('Q-Q Plot of Residuals (Norm of errs)')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    st.pyplot(fig3)

# Defining a function to create Post Estimation diagnostics
@st.cache_resource
def post_estimation_diagnostics(y_variable, x_variable, residuals, _results, name):
    # Running the diagnostics
    name = f"Post Estimation Diagnostics for {name}"
    st.markdown(f"<h2 style = 'text-align: center;'>{name}</h2>", unsafe_allow_html = True)


    # Creating an empty table 
    dictionary = {
        "Heteroscedasticity test 1" : ["","","","",""],
        "Heteroscedasticity test 2" : ["","","","",""],
        "Heteroscedasticity test 3" : ["","","","",""],
        "Autocorrelation test 1" : ["","","","",""],
        "Autocorrelation test 2" : ["","","","",""],
        "Autocorrelation test 3" : ["","","","",""],
        "Autocorrelation test 4" : ["","","","",""],
        "Autocorrelation test 5" : ["","","","",""],
        "Normality test" : ["","","","",""],
        "Linearity test" : ["","","","",""]
    } 
    df = pd.DataFrame(dictionary)
    df = df.transpose()
    df.columns = ["Ho","Test", "F-Statistic", "P-Value", "Verdict"]

    # Runninng all the tests

    # Goldfeld - Quant test
    goldfeld_quandt_test = het_goldfeldquandt(y_variable, x_variable)

    # White test
    white_test = het_white(residuals, _results.model.exog)

    # Arch Test
    arch_test = het_arch(residuals)

    # Durbin Watson test
    durbin_watson_t = durbin_watson(residuals)

    # Breusch - Godfrey test
    bg_test_1 = acorr_breusch_godfrey(_results, nlags = 2)
    bg_test_2 = acorr_breusch_godfrey(_results, nlags = 3)
    bg_test_3 = acorr_breusch_godfrey(_results, nlags = 4)
    bg_test_4 = acorr_breusch_godfrey(_results, nlags = 5)

    # Jarque bera test
    jb_test = jarque_bera(residuals)

    # Ramsey reset test
    ramsey_test = linear_reset(_results, power = 3, use_f = True) # performs F-test. Can also do a T-test

    # Doing the table
    df.iloc[0,0] = "Homoscedasticity"
    df.iloc[0,1] = "Goldfeld - Quant test" 
    df.iloc[0,2] = round(goldfeld_quandt_test[0], 3)
    df.iloc[0,3] = round(goldfeld_quandt_test[1], 3) 

    df.iloc[1,0] = "Homoscedasticity"
    df.iloc[1,1] = "White test"
    df.iloc[1,2] = round(white_test[0], 3)
    df.iloc[1,3] = round(white_test[1], 3) 

    df.iloc[2,0] = "Homoscedasticity"
    df.iloc[2,1] = "Arch LM test"
    df.iloc[2,2] = round(arch_test[0], 3)
    df.iloc[2,3] = round(arch_test[1], 3)

    df.iloc[3,0] = "F.O. Autocorrelation"
    df.iloc[3,1] = "Durbin Watson test"
    df.iloc[3,2] = round(durbin_watson_t, 3)
    df.iloc[3,3] = np.nan
    df.iloc[3,4] = np.nan

    df.iloc[4,0] = "2.O. Autocorrelation"
    df.iloc[4,1] = "Breusch-Godfrey test"
    df.iloc[4,2] = round(bg_test_1[0], 3)
    df.iloc[4,3] = round(bg_test_1[1], 3)

    df.iloc[5,0] = "3.O. Autocorrelation"
    df.iloc[5,1] = "Breusch-Godfrey test"
    df.iloc[5,2] = round(bg_test_2[0], 3)
    df.iloc[5,3] = round(bg_test_2[1], 3)

    df.iloc[6,0] = "4.O. Autocorrelation"
    df.iloc[6,1] = "Breusch-Godfrey test"
    df.iloc[6,2] = round(bg_test_3[0], 3)
    df.iloc[6,3] = round(bg_test_3[1], 3)

    df.iloc[7,0] = "5.O. Autocorrelation"
    df.iloc[7,1] = "Breusch-Godfrey test"
    df.iloc[7,2] = round(bg_test_4[0], 3)
    df.iloc[7,3] = round(bg_test_4[1], 3)

    df.iloc[8,0] = "Errors are normal"
    df.iloc[8,1] = "Jarque Bera test"
    df.iloc[8,2] = round(jb_test[0], 3)
    df.iloc[8,3] = round(jb_test[1], 3)

    df.iloc[9,0] = "Func Form is Correct"
    df.iloc[9,1] = "Ramsey Reset Test"
    df.iloc[9,2] = round(ramsey_test.fvalue, 3)
    df.iloc[9,3] = round(ramsey_test.pvalue, 3)

    for i in itertools.chain(range(0, 3), range(4, 10)):
        if df.iloc[i,3] < 0.05:
            df.iloc[i,4] = "Reject Ho at 5%" 
        elif df.iloc[i,3] < 0.01:
            df.iloc[i,4] = "Reject Ho at 1%" 
        else: 
            df.iloc[i,4] = "Do not reject Ho"
    
    return df

# Defining a function to draw exogenous and endogenous variables 
@st.cache_resource
def variabs(regressand, regressors, nov, dates = None):

    # Creating a list that contains the names of variables
    regressor_names = regressors.columns.tolist()
    regressand_name = regressand.columns.tolist()

    # Creating a universal figure size
    figsize = (12,4)

    # Adjusting the dates
    if dates is not None:
        dates = pd.to_datetime(dates)
        index = dates
    else:
        index = np.arange(regressand.shape[0])

    # Ensuring that index has the same index as regressand
    # the need for this occurs only if lengths of index and regressand are not equal
    if len(index) != len(regressors):
        index_to_copy = regressors.index
        index = [index[i] for i in index_to_copy]
    

    # Creating a figure for Y variable
    fig1, axes1 = plt.subplots(1, 2, figsize = figsize) 

    # Visual of Y
    axes1[0].plot(index, regressand)
    axes1[0].set_title(f"Plot of {regressand_name[0]}")
    #axes1[0].set_xlabel("Time")
    axes1[0].set_ylabel("Value")
    axes1[0].set_xticks(index)
    n_to_plot = len(index) / 10
    n_to_plot = int(round(n_to_plot, 0))
    every_other_date = index[::n_to_plot]
    axes1[0].set_xticks(every_other_date)
    axes1[0].tick_params(axis='x', rotation=45)

    axes1[1].hist(regressand, bins = 50)
    axes1[1].set_title(f"Histogram of {regressand_name[0]}")
    axes1[1].set_xlabel("Intervals")
    axes1[1].set_ylabel("Frequency")
    axes1[1].set_xticklabels([])
    st.markdown("<h2 style = 'text-align: center;'>Endogenous variable</h2>", unsafe_allow_html = True)

    fig1.tight_layout()
    st.pyplot(fig1)

    # Creating a figure for X variable
    fig2, axes2 = plt.subplots(nov - 1, 2, figsize = (12, 4 * (nov - 1))) # only considers constant

    # Plotting the plots of the exogenous variables and the histograms
    if nov > 2:
        for i in range(nov - 1): # i have four variables but i will draw only three. i is [0,1,2]
            axes2[i,0].plot(index, regressors.iloc[0:,i+1])
            axes2[i,0].set_title(f"Plot of {regressor_names[i + 1]}")
            #axes2[i,0].set_xlabel("Time")
            axes2[i,0].set_ylabel("Value")
            n_to_plot = len(index) / 10
            n_to_plot = int(round(n_to_plot, 0))
            every_other_date = index[::n_to_plot]
            axes2[i,0].set_xticks(every_other_date)
            axes2[i,0].tick_params(axis='x', rotation=45)
            
            axes2[i,1].hist(regressors.iloc[0:,i+1], bins = 50)
            axes2[i,1].set_title(f"Histogram of {regressor_names[i + 1]}")
            axes2[i,1].set_xlabel("Intervals")
            axes2[i,1].set_ylabel("Frequency")
    elif nov == 2:
        axes2[0].plot(index, regressors.iloc[0:,1])
        axes2[0].set_title(f"Plot of {regressor_names[1]}")
        #axes2[0].set_xlabel("Time")
        axes2[0].set_ylabel("Value")
        n_to_plot = len(index) / 10
        n_to_plot = int(round(n_to_plot, 0))
        every_other_date = index[::n_to_plot]
        axes2[0].set_xticks(every_other_date)
        axes2[0].tick_params(axis='x', rotation=45)
            
        axes2[1].hist(regressors.iloc[0:,1], bins = 50)
        axes2[1].set_title(f"Histogram of {regressor_names[1]}")
        axes2[1].set_xlabel("Intervals")   
        axes2[1].set_ylabel("Frequency") 
    
    if regressors.shape[1] > 2:     
        st.markdown("<h2 style = 'text-align: center;'>Exogenous variables</h2>", unsafe_allow_html = True)
    else:
        st.markdown("<h2 style = 'text-align: center;'>Exogenous variable</h2>", unsafe_allow_html = True)

    fig2.tight_layout()
    st.pyplot(fig2)

# Defining OLS with simple errors
@st.cache_resource
def simple_ols(regressand, regressors):
    model = sm.OLS(regressand, regressors)
    results = model.fit()
    return results, model

# Defining OLS with robust errors
@st.cache_resource
def robust_ols(regressand, regressors):
    model = sm.OLS(regressand, regressors)
    results = model.fit(cov_type='HC1')
    return results, model

# Chow test function
@st.cache_resource
def chow_test(split_point, regressand, regressors, nobs):

    x1, x2 = regressors.iloc[:split_point, :], regressors.iloc[split_point:, :]
    y1, y2 = regressand.iloc[:split_point], regressand.iloc[split_point:]

    # Conducting three regression
    model_1 = sm.OLS(regressand, regressors)  # full model
    model_2 = sm.OLS(y1, x1) # before split point 
    model_3 = sm.OLS(y2, x2) # after split point

    # Fitting the models
    results_1 = model_1.fit()
    results_2 = model_2.fit()
    results_3 = model_3.fit()

    # Conducting the actual test
    rss_1 = results_1.ssr
    rss_2 = results_2.ssr
    rss_3 = results_3.ssr
    k = x1.shape[1]
    T = nobs

    # test_statistic = (((rss_1 - (rss_2 + rss_3))) / (rss_2 + rss_3)) / ((T - 2 * k) / k)
    # p_value = 1 - stats.f.cdf(test_statistic, k, T - 2 * k)

    # Chow test F-statistic
    numerator = (rss_1 - (rss_2 + rss_3)) / k
    denominator = (rss_2 + rss_3) / (T - 2 * k)
    
    # F-statistic
    test_statistic = numerator / denominator

    # P-value from the F-distribution
    p_value = 1 - stats.f.cdf(test_statistic, k, T - 2 * k)

    if p_value < 0.05:
        result = "reject the Ho. There is a structural break."
    else:
        result = "fail to reject Ho. There is no structural break."

    text = f"The Chow Test F-statistic is {round(test_statistic,3)}. \nThe p-value is {round(p_value,3)}. \nTherefore, we {result}"
    text = str(text)
    # Displaying text with new lines
    st.text(text)

# Function to add lagged variables of y
@st.cache_resource
def add_lag(regressand, regressors, nol):

    # Creating a dataframe that contains everything
    df = pd.DataFrame()

    # Retrieving the name of the endogenous variable
    column_name = regressand.columns[0]
    df[column_name] = regressand
    df = pd.concat([df, regressors], axis = 1)

    # Creating a set of lagged variables
    for i in range(1, nol + 1):
        df[f"{column_name} lagged by {i}"] = df[column_name].shift(i)

    # Cleaning the NaNs
    df = df.dropna()

    # Creating a new df for x
    new_x = df.drop(columns = [column_name])

    # Creating a new df for y
    new_y = df[column_name]

    # Turning the new_y into a dataframe too
    new_y = pd.DataFrame(new_y)

    return new_y, new_x

# Function to clean the column with the dates
@st.cache_resource
def data_converter(date_col):
    new_dates = []
    for value in date_col:
        new_dates.append(str(value).split(" ")[0])
    return new_dates

# Function to make a table with a loss function 
@st.cache_resource
def loss_function(out_of_sample_data, forecasts, column_name, theil_number):
    mae_str = mean_absolute_error(out_of_sample_data, forecasts)
    mape_str = mean_absolute_percentage_error(out_of_sample_data, forecasts)
    rmse_str = np.sqrt(mean_squared_error(out_of_sample_data, forecasts))

    str_forecast_loss_functions = pd.DataFrame({
        "MAE": [mae_str],
        "MAPE": [mape_str],
        "RMSE": [rmse_str],
        "Theil's U": [theil_number]
        }).T
    str_forecast_loss_functions.columns = [column_name]
    return str_forecast_loss_functions

# Function to calsulate the theils u
@st.cache_resource
def theils_u(actual, forecast):

    actual = np.array(actual)
    forecast = np.array(forecast)

    # Calculate the numerators
    numerator = np.sqrt(np.mean(((forecast[1:] / actual[:-1]) - (actual[1:] / actual[:-1])) ** 2))

    # Calculate the denominator
    denominator = np.sqrt(np.mean(((actual[1:] / actual[:-1]) - 1) ** 2))

    # Theil's U calculation
    theil_u_stat = numerator / denominator

    return theil_u_stat

st.markdown("<h1 style='text-align: center;'>Easy Regress</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>File Upload</h1>", unsafe_allow_html=True)
st.markdown("---")
st.sidebar.markdown("<h1 style='text-align: center;'>Activity log</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
index_1 = 0

# Uploading a file
file = st.file_uploader("Upload your file",
                         type = ["xlsx", "csv"])

# Printing the uploaded data
if file is not None:
    st.sidebar.markdown("<h1 style='text-align: center;'>General info</h1>", unsafe_allow_html=True)

    # Getting the file name
    file_name = file.name

    # Get the file extension
    file_extension = file.name.split('.')[-1]

    # Updating the sidebar to reflect on loaded data
    index_1 += 1
    st.sidebar.markdown(f"{index_1}. Data file: **{file_name}** was uploaded.")
    
    # Read the file based on its extension
    if file_extension == 'csv':
        data = pd.read_csv(file)
    elif file_extension == 'xlsx':
        data = pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a .csv or .xlsx file.")

    # Code to show the data if the user wants
    if st.checkbox("Tick to show the uploaded file."):
        st.write(data)   

    # Extracting variables
    variables = data.columns.tolist()

    # Updating the sidebar to reflect on loaded data
    variables_str = ", ".join(variables)

    if len(variables) == 1: 
        index_1 += 1
        st.sidebar.markdown(f"{index_1}. The variable contained in the data file is: **{variables_str}**.")
    else:
        index_1 += 1
        st.sidebar.markdown(f"{index_1}. The variables contained in the data file are: **{variables_str}**.")

    # Asking the user to specify the Xs, Y, and a date column
    dates_checkbox = st.checkbox("Tick if there is a date column in your file.")
    if dates_checkbox:
        st.warning("It is recommended that the date variable is of a DD/MM/YYYY format.")
        dates = st.selectbox("Select the date variable",
                             options = variables)
        
        # Ensuring the dates could not be selected as a regressor or regressand 
        variables = [var for var in variables if var != dates]

        dates = data[dates]
        dates_str = dates.name

        # Updating the sidebar to reflect on loaded data
        index_1 += 1
        st.sidebar.markdown(f"{index_1}. The chosen date variable is: **{dates_str}**.")

        dates = data_converter(dates)
        dates = pd.Series(dates).copy() # this is the bit that can be deleted
        
    else:
        dates = None
    # Selecting the Y variable
    y_var = st.selectbox("Please select endogenous variable",
                         options = variables)

    # Extracting y variable
    if y_var:    
        regressand = data[y_var]
        regressand_str = regressand.name
        #st.write(type(regressand[0]))

        # If regressand is a date, then it will be string. I dont need string
        # Checking if the regressand if a string
        allow_plot = True
        allow_analysis = True
        if pd.api.types.is_string_dtype(regressand):
            st.warning(f"The {regressand_str} variable cannot be selected as an independent variable!")
            allow_plot = False
            allow_analysis = False

        # Updating the sidebar to reflect on loaded data
        index_1 += 1
        st.sidebar.markdown(f"{index_1}. The chosen regressand is: **{regressand_str}**.")

        regressand = pd.DataFrame(regressand)

        # Creating a copy for the ARIMA bit
        arima_regressand = regressand.copy()

        # Creating a copy for the ARIMA bit
        auto_arima_regressand = regressand.copy()

        # Creating a copy of regressand for the Chow test
        orig_regressand = regressand.copy()
        orig_regressand_to_plot = regressand.copy()

        # Plotting the Y variable
        if allow_plot:
            if dates_checkbox:
                orig_dates = pd.to_datetime(dates)
                orig_regressand_to_plot.index = orig_dates
                st.line_chart(orig_regressand_to_plot)
            else:
                st.line_chart(orig_regressand_to_plot)            

        # Extracting the number of observations and the number of variables
        nobs = len(regressand)

    # Removing the selected Y variable from the list of available X variables
    x_var_options = [var for var in variables if var != y_var]

    # Selecting the X variables
    x_vars = st.multiselect("Please select all exogenous variables",
                            options = x_var_options)       
    
    # Creating a separate dataframe with only needed x variables
    if x_vars:
        regressors = data[x_vars]
        regressors_str = ", ".join(regressors)
        index_1 += 1

        # Updating the sidebar to reflect on loaded data
        if len(variables) == 1: 
            st.sidebar.markdown(f"{index_1}. The chosen regressor is: **{regressors_str}**.")
        else:
            st.sidebar.markdown(f"{index_1}. The chosen regressors are: **{regressors_str}**.")

        regressors = sm.add_constant(regressors)

        # Creating a copy of regressors for the Chow test
        orig_regressors = regressors.copy()
    
        # Extracting the numbers of variables
        nov = regressors.shape[1] 

    st.sidebar.markdown("---")
    
# STRUCTURAL FORECASTING
    if y_var and x_vars:
        if allow_analysis:
            structural_analysis = st.checkbox("Tick to start structural analysis.")
            if structural_analysis:
                st.markdown("---")
                st.markdown("<h1 style='text-align: center;'>Structural Model Analysis</h1>", unsafe_allow_html=True)
                # Updating the sidebar to reflect on loaded data
                st.sidebar.markdown("<h1 style='text-align: center;'>Structural Analysis</h1>", unsafe_allow_html=True)

                # Plotting all the variables
                st.markdown("<h1 style='text-align: center;'>Visual inspection of all variables</h1>", unsafe_allow_html=True)
                if dates_checkbox:
                    variabs(regressand = regressand, regressors = regressors, nov = nov, dates = dates)
                else:
                    variabs(regressand = regressand, regressors = regressors, nov = nov)

                st.markdown("---")

                # An option to take logs of both the regressand and the regressor
                str_logs = st.checkbox(f"Tick to take **logs** of {regressand_str}, {regressors_str}.", key = "logs_1")
                if str_logs:
                        
                    # Updating the sidebar to reflect on loaded data
                    index_1 += 1
                    st.sidebar.markdown(f"{index_1}. The **logs** of {regressand_str}, {regressors_str} have been taken.")

                    st.markdown("<h1 style='text-align: center;'>Visual of log data</h1>", unsafe_allow_html=True)
                    regressand = np.log(regressand).copy()
                    regressors = np.log(regressors).copy()

                    # Ensuring the first column of regressors is 1
                    regressors.iloc[:, 0] = 1

                    # Check if regressand and regressors have any None values
                    has_missing_1 = regressand.isnull().values.any() # returns true of false
                    has_missing_2 = regressors.isnull().values.any() # returns true of false

                    columns_with_missing = regressors.isnull().any() # this returned that weird table
                    columns_with_missing_names = regressors.columns[columns_with_missing] # this will get the names of the variables with None values
                    columns_with_nones = ', '.join(columns_with_missing_names) # this just contains the names of the variable                    

                    # Updating the sidebar to reflect on loaded data
                    if has_missing_1 and has_missing_2:  
                        regressand = regressand.dropna()
                        regressors = regressors.dropna()    
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. After taking logs, **{regressand_str}, {columns_with_nones}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)

                    elif has_missing_1:
                        regressand = regressand.dropna()
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. After taking logs, **{regressand_str}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                        
                    elif has_missing_2:
                        regressors = regressors.dropna()
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. After taking logs, **{columns_with_nones}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)

                    # In case no Nones occur
                    else:
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. After taking logs, **no variable** had None values.<br>The number of observations is: **{previous_nobs}**.", unsafe_allow_html=True)

                    # Ensuring that both the regressors and the regressand have the same index
                    regressand, regressors = regressand.align(regressors, join='inner', axis=0) # this worked very well

                    # Resetting the index of the regressand and the regressors
                    regressand = regressand.reset_index(drop=True)
                    regressors = regressors.reset_index(drop=True)

                    # Visualising the updated variables
                    if dates_checkbox:
                        variabs(regressand = regressand, regressors = regressors, nov = nov, dates = dates)
                    else:
                        variabs(regressand = regressand, regressors = regressors, nov = nov)
                                    
                st.markdown("---")

                # An option to take differences of regressor and the regressand 
                if str_logs:
                    take_difference1 = st.checkbox(f"Tick to take a **difference** of **log** {regressand_str}, {regressors_str}.")
                else:
                    take_difference1 = st.checkbox(f"Tick to take a **difference** of {regressand_str}, {regressors_str}.")

                if take_difference1:

                    # Taking the actual differences and dropping Nones
                    regressand = regressand.diff().dropna().copy()
                    regressors = regressors.diff().dropna().copy()

                    # Resetting the index of the regressand and the regressors
                    regressand = regressand.reset_index(drop=True)
                    regressors = regressors.reset_index(drop=True)

                    # Retrieving the new number of observations
                    if str_logs:
                        previous_nobs = updated_nobs # the previous new becomes old
                    else:
                        previous_nobs = nobs
                    updated_nobs = len(regressand) # the new new becomes new
                    
                    # Updating the sidebar to reflect on loaded data

                    index_1 += 1

                    if str_logs:
                        st.sidebar.markdown(f"{index_1}. The difference of the **log {regressand_str}, {regressors_str}** has been taken.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                        st.markdown(f"<h1 style = 'text-align: center;'>Plot of differenced log {regressand_str}, {regressors_str}</h1>", unsafe_allow_html = True)    
                    else: 
                        st.sidebar.markdown(f"{index_1}. The difference of the **{regressand_str}, {regressors_str}** has been taken.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                        st.markdown(f"<h1 style = 'text-align: center;'>Plot of differenced {regressand_str}, {regressors_str}</h1>", unsafe_allow_html = True)
                    
                    # Ensuring the first columns is 1 
                    regressors.iloc[:, 0] = 1

                    if dates_checkbox:
                        variabs(regressand = regressand, regressors = regressors, nov = nov, dates = dates)
                    else:
                        variabs(regressand = regressand, regressors = regressors, nov = nov)

                st.markdown("---")

                # Option to enter lagged variables if any
                nol = st.number_input("Enter the number of lagged variables you need.", 
                                        min_value = 0,
                                        step = 1)
                if nol:
                    regressand, regressors = add_lag(regressand = regressand,
                                                    regressors = regressors,
                                                    nol = nol)
                    
                    # Resetting the index of the regressand and the regressors
                    regressand = regressand.reset_index(drop=True)
                    regressors = regressors.reset_index(drop=True)
                    
                    # Retrieving the new number of observations
                    if str_logs and take_difference1:
                        previous_nobs = updated_nobs 
                    elif str_logs:
                        previous_nobs = updated_nobs
                    elif take_difference1:
                        previous_nobs = updated_nobs
                    else:
                        previous_nobs = nobs
                    updated_nobs = len(regressand)
        
                    # Updating the sidebar to reflect on loaded data
                    index_1 += 1
                    if str_logs:  
                        st.sidebar.markdown(f"{index_1}. The number of lagged variables of {regressand_str} is **{nol}**.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                    else:
                        st.sidebar.markdown(f"{index_1}. The number of lagged variables of {regressand_str} is **{nol}**.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                else:
                    nol = 0

                if st.checkbox("Tick to show regressors."):
                    st.write(regressors)
                if st.checkbox("Tick to show regressand."):
                    st.write(regressand)
                st.markdown("---")

                # Running a regression on the entire data

                if st.checkbox("Tick if you want robust errors in the model.", key = "model_1"):
                    fitted_model_str_whole, model_str_whole = robust_ols(regressand = regressand, regressors = regressors)
                else:
                    fitted_model_str_whole, model_str_whole = simple_ols(regressand = regressand, regressors = regressors)

                # Retrieving fitted values and residuals
                fitted_values_str_whole = fitted_model_str_whole.fittedvalues
                residuals_str_whole = fitted_model_str_whole.resid

                # Printing the Regression Output
                st.markdown("<h2 style = 'text-align: center;'>Structural regression output</h2>", unsafe_allow_html = True)
                st.write(fitted_model_str_whole.summary())
                st.markdown("---", unsafe_allow_html = True)

                # Running the post estimation diagnostics
                name2 = "structural model (entire data)"
    
                PEDs1 = post_estimation_diagnostics(y_variable = regressand, # here by the way, the regressand is not the in sample one
                                                x_variable = regressors,
                                                residuals = residuals_str_whole,
                                                _results = fitted_model_str_whole,
                                                name = name2)

                st.write(PEDs1)
                st.markdown("---")

                # Checking for the Gauss Markov Assumptions
                GMs1 = gauss_markov(fitted_values = fitted_values_str_whole,
                                    residuals = residuals_str_whole)
                st.markdown("---")

                # Showing the correlation matrix for the regressors
                if st.checkbox("Tick to show correlation matrix of regressors.", key = "Entire model matrix"):
                    st.markdown("<h2 style = 'text-align: center;'>Coefficient Correlation Matrix</h2>", unsafe_allow_html = True)
                    regressors_no_constant = regressors.drop(columns = ["const"])
                    correlation_matrix = regressors_no_constant.corr()
                    st.write(correlation_matrix)
                st.markdown("---")
                        
                # Chow test
                if st.checkbox("Tick to perform a Chow test."):
                    st.write(f"**Please, note that the Chow test is run on the entire dataset.**")
                    st.line_chart(orig_regressand_to_plot)
                    if dates_checkbox:  
                        try:
                            dates = [datetime.datetime.strptime(date, "%d/%m/%Y").date() for date in dates] # original
                            selected_date = st.select_slider("Select a date", options = dates)      
                            try:          
                                chow_index_1 = dates.index(selected_date)
                                chow1 = chow_test(split_point = chow_index_1,
                                                regressand = orig_regressand, 
                                                regressors = orig_regressors,
                                                nobs = nobs)  # i am using the entire dataset so nobs remains as it is
                                #st.write(chow1)
                            except ValueError:
                                st.write(f"{selected_date} is not in the list.")
                        except ValueError:
                            st.warning("There has been an error with the dates format. Please, go up and **unselect** the date variable.")

                    else:
                        split_point_1 = st.number_input("Enter the date for the Chow Test",
                                                        key = "chow_1", 
                                                        step = 1, 
                                                        min_value = 1,
                                                        max_value = len(regressand))
                        if split_point_1:
                            chow1 = chow_test(split_point = split_point_1,
                                            regressand = orig_regressand, 
                                            regressors = orig_regressors,
                                            nobs = nobs)
                            #st.write(chow1)
                        else:
                            st.write("Please enter a valid split point to conduct the Chow test.")
        
                st.markdown("---")

                # Asking the user to specify the number to break the data
                first_forecast = st.checkbox("Tick to perform in sample and out of sample forecating ability of the model.")
                if first_forecast:
                    index_1 += 1
                    in_sample_proportion = st.number_input("Proportion for in-sample data", step = 0.05, value = 0.7)
    
                    st.markdown("---")
                    if in_sample_proportion:
                        if not "updated_nobs" in globals():
                            updated_nobs = nobs
                    
                        out_of_sample_proportion = 1 - in_sample_proportion
                        in_sample_nobs = round(updated_nobs * in_sample_proportion) 

                        # Updating the sidebar to reflect on loaded data
                        st.sidebar.markdown(f"{index_1}. The in sample proportion is selected at **{round(in_sample_proportion * 100)}%**. Out of sample proportion is selected at **{round(100 - in_sample_proportion * 100)}%**.", unsafe_allow_html=True) 
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. The in sample number of observations is: **{in_sample_nobs} (before data splitting: {updated_nobs})**. The number of observations to forecast is: **{updated_nobs - in_sample_nobs}**.", unsafe_allow_html=True) 

                        # Now will break all data into in sample and out of sample data
                        # Creating in sample Y and out of sample Y
                        in_sample_y, out_of_sample_y = regressand.iloc[:in_sample_nobs].copy(), regressand.iloc[in_sample_nobs:].copy()

                        # Creating in sample X and out of sample X
                        in_sample_x, out_of_sample_x = regressors.iloc[:in_sample_nobs, :].copy(), regressors.iloc[in_sample_nobs:, :].copy()

                        # Creating a model to create the next nobs_to_predict variables 
                        if st.checkbox("Tick if you want robust errors in the model.", key = "model_2"):
                            fitted_model, model = robust_ols(regressand = in_sample_y, regressors = in_sample_x)
                        else:
                            fitted_model, model = simple_ols(regressand = in_sample_y, regressors = in_sample_x)

                        # Retrieving fitted values and residuals
                        fitted_values_str = fitted_model.fittedvalues
                        residuals_str = fitted_model.resid

                        # Printing the Regression Output
                        st.markdown("<h2 style = 'text-align: center;'>Structural regression output</h2>", unsafe_allow_html = True)
                        st.write(fitted_model.summary())
                        st.markdown("---", unsafe_allow_html = True)

                        # Extracting the real next nobs_to_predict explanatory variables
                        # out_of_sample_x is the thing i am looking for

                        # FORECASTING the next nobs_to_predict Y variable
                        str_forecasts = fitted_model.predict(out_of_sample_x)
                    
                        # Now will compare the REAL NUMBERS with PREDICTIONS only if regressors are specified
                        if regressors.shape[1] > 1:

                            # Plotting the actual Y together with the forecast
                            st.markdown("<h1 style='text-align: center;'>Comparing actual data with out of sample forecasts</h1>", unsafe_allow_html=True)
                            regressand_str_2 = regressand.copy()
                            regressand_str_2["str_forecasted_values"] = [None]*len(in_sample_y) + list(str_forecasts)
                            fig1, ax = plt.subplots()
                
                            # Plotting exercise
                            if str_logs and take_difference1:   
                                label_name_4 = f"Differenced log {regressand_str}"
                            elif str_logs:
                                label_name_4 = f"Log {regressand_str}"                             
                            elif take_difference1:
                                label_name_4 = f"Differenced {regressand_str}"
                            else:
                                label_name_4 = f"{regressand_str}" 
                         
                            plotting_exercise(regressand = regressand_str_2,
                                              label_name = label_name_4,
                                              dates_checkbox = dates_checkbox,
                                              dates = dates)
                                                        
                            st.markdown("---")

                            # Calculating the loss functions 
                            st.markdown("<h1 style='text-align: center;'>Estimating the loss functions from the structural model</h1>", unsafe_allow_html=True)
                            name1 = "Structural model"
                            theils_1 = theils_u(actual = out_of_sample_y, forecast = str_forecasts)
                            loss_functions_1 = loss_function(out_of_sample_data = out_of_sample_y, forecasts = str_forecasts, column_name = name1, theil_number = theils_1)
                            st.write(loss_functions_1)
                            st.markdown("---")

                        # Running the post estimation diagnostics
                        if st.checkbox("Tick to show the Post Estimation Diagnostics for the forecast model."):
                            PEDs2 = post_estimation_diagnostics(y_variable = in_sample_y, # here by the way, the regressand is not the in sample one
                                                            x_variable = in_sample_x,
                                                            residuals = residuals_str,
                                                            _results = fitted_model,
                                                            name = name1)

                            st.write(PEDs2)
                            st.markdown("---")

                            # Checking for the Gauss Markov Assumptions
                            GMs2 = gauss_markov(fitted_values = fitted_values_str,
                                                residuals = residuals_str)
                        st.markdown("---")

                        # Showing the correlation matrix for the regressors
                        if st.checkbox("Tick to show correlation matrix of regressors.", key = "Forecast model matrix"):
                            st.markdown("<h2 style = 'text-align: center;'>Coefficient Correlation Matrix</h2>", unsafe_allow_html = True)
                            regressors_no_constant = regressors.drop(columns = ["const"])
                            correlation_matrix = regressors_no_constant.corr()
                            st.write(correlation_matrix)
                st.markdown("---")

# Manual ARIMA

#     if y_var:
#         if allow_analysis:
#             # arima_regressand is just a copy of the original data for the ARIMA purposes
#             arima_previous_nobs = len(arima_regressand)
#             do_arima = st.checkbox("Tick to start ARIMA analysis.") 
#             if do_arima:
#                 st.markdown("---")
#                 # Updating the sidebar to reflect on loaded data t
#                 st.sidebar.markdown("---")
#                 st.sidebar.markdown("<h1 style='text-align: center;'>ARIMA  Manual Analysis</h1>", unsafe_allow_html=True)
#                 st.markdown("<h1 style='text-align: center;'>ARIMA Analysis</h1>", unsafe_allow_html=True)
#                 st.markdown("<h1 style='text-align: center;'>Box Jenkins Methodology</h1>", unsafe_allow_html=True)
#                 st.markdown("<h1 style='text-align: center;'>Step 1: Visual Inspection</h1>", unsafe_allow_html=True)

#                 # Plottimg exercise
#                 label_name_8 = f"{regressand_str}"
#                 plotting_exercise(regressand = arima_regressand,
#                                   label_name = label_name_8,
#                                   dates_checkbox = dates_checkbox,
#                                   dates = dates)

#                 st.markdown("---")

#                 # Option to take the logs of data
#                 arima_logs = st.checkbox(f"Tick to take **logs** of {regressand_str}.", key = "logs_2")
#                 if arima_logs:

#                     # Updating the sidebar to reflect on loaded data
#                     index_1 += 1
#                     st.sidebar.markdown(f"{index_1}. The **logs** of {regressand_str} have been taken.")

#                     st.markdown("<h1 style = 'text-align: center;'>Log data</h1>", unsafe_allow_html = True)    
#                     arima_regressand = np.log(arima_regressand).copy()

#                     # Ensuring the None values are gone
#                     has_missing_3 = arima_regressand.isnull().values.any() # returns true of false
#                     if has_missing_3:
#                         arima_regressand = arima_regressand.dropna()

#                         # Resetting the index of the arima_regressand and getting the last index of the arima_regressand
#                         arima_regressand = arima_regressand.reset_index(drop=True)
                        
#                         # Extracting the new number of observations IF there are Nones
#                         arima_updated_nobs = len(arima_regressand)

#                         # Making a comment in the sidebar
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. After taking logs, **{regressand_str}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{arima_updated_nobs} (old: {arima_previous_nobs})**.", unsafe_allow_html=True)
#                     else:
#                         # Making a comment in the sidebar
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. After taking logs, **{regressand_str}** had no None values.<br>The number of observations is still: **{arima_previous_nobs})**.", unsafe_allow_html=True)

#                         # Extracting the new number of observations IF there are Nones
#                         arima_updated_nobs = len(arima_regressand)

#                     # Plotting exercise
#                     label_name_1 = f"Log {regressand_str}"
                    
#                     # Plotting exercise
#                     plotting_exercise(regressand = arima_regressand,
#                                       label_name = label_name_1,
#                                       dates_checkbox = dates_checkbox,
#                                       dates = dates)
#                 st.markdown("---")

#                 # Breaking the data into in sample and out of sample data
#                 second_forecast = st.checkbox("Tick to perform in sample and out of sample forecating ability of the model.", key = "arima split 1")
#                 if second_forecast:

#                     arima_in_sample_proportion = st.number_input("Proportion for in-sample data", step = 0.05, value = 0.7, key = "arima split 2")
                   
#                     if arima_in_sample_proportion:
#                         if not "arima_updated_nobs" in globals():
#                             arima_updated_nobs = nobs
                        
#                         out_of_sample_proportion_2 = 1 - arima_in_sample_proportion
#                         arima_in_sample_nobs = round(arima_updated_nobs * arima_in_sample_proportion) 

#                         # Updating the sidebar to reflect on loaded data
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. The in sample proportion is selected at **{round(arima_in_sample_proportion * 100)}%**. Out of sample proportion is selected at **{round(100 - arima_in_sample_proportion * 100)}%**.", unsafe_allow_html=True) 
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. The in sample number of observations is: **{arima_in_sample_nobs} (before data splitting: {arima_updated_nobs})**. The number of observations to forecast is: **{arima_updated_nobs - arima_in_sample_nobs}**.", unsafe_allow_html=True) 

#                         # Splitting the data into in sample and out of sample data
#                         in_sample_arima_y, out_of_sample_arima_y = arima_regressand.iloc[:arima_in_sample_nobs].copy(), arima_regressand.iloc[arima_in_sample_nobs:].copy()

#                         # Checking for stationarity
#                         st.markdown("---")
#                         if arima_logs:
#                             st.markdown(f"<h1 style = 'text-align: center;'>Checking log {regressand_str} for stationarity</h1>", unsafe_allow_html = True)    
#                         else:
#                             st.markdown(f"<h1 style = 'text-align: center;'>Checking {regressand_str} for stationarity</h1>", unsafe_allow_html = True)

#                         # Conducting the first round of Augmented Dickey Fuller test
#                         st.markdown("<h1 style = 'text-align: center;'>Augmented Dickey Fuller test test</h1>", unsafe_allow_html = True)
#                         adf_test_1 = adfuller(in_sample_arima_y)

#                         if adf_test_1[1] > 0.05:
#                             st.write("")
#                             st.write(f"p value is **{round(adf_test_1[1], 3)}**. Hence, There is not enough information to reject Ho. \n**We have evidence of a unit root**.")
#                             st.write(f"**Data is not stationary**.")
#                             st.markdown("---")           
#                         else:
#                             st.write("")
#                             st.write(f"p value is **{round(adf_test_1[1], 3)}**. Hence, we have enough evidence to reject Ho. \nThere is **no evidence of a unit root**.")
#                             st.write(f"**Data is stationary**.")
#                             st.markdown("---")

#                         # Plotting the first round of correlograms
#                         if arima_logs:
#                             st.markdown(f"<h1 style = 'text-align: center;'>Correlograms for log {regressand_str}</h1>", unsafe_allow_html = True)    
#                         else:
#                             st.markdown(f"<h1 style = 'text-align: center;'>Correlograms for {regressand_str}</h1>", unsafe_allow_html = True)


#                         # Plotting ACF and PACF
#                         acf_pacf(series1 = in_sample_arima_y, series2 = in_sample_arima_y)
#                         st.markdown("---")

#                         # Option to take difference of data
#                         if arima_logs:
#                             take_difference2 = st.checkbox(f"Tick to take a **difference** of **log** {regressand_str}.")
#                         else:
#                             take_difference2 = st.checkbox(f"Tick to take a **difference** of {regressand_str}.")

#                         if take_difference2:

#                             # Updating the sidebar to reflect on loaded data
#                             index_1 += 1
#                             st.sidebar.markdown(f"{index_1}. Please, note that the differences are taken **solely** for visualisation purposes. The data will **not** be differenced in the stage of manual ARIMA estimation", unsafe_allow_html = True)   

#                             # Taking the difference
#                             in_sample_arima_y_diff = in_sample_arima_y.diff().dropna().copy()

#                             # Resetting the index of in_sample_arima_y_diff
#                             in_sample_arima_y_diff = in_sample_arima_y_diff.reset_index(drop=True)

#                             # Extracting the new number of observations
#                             if arima_logs:
#                                 arima_previous_nobs = arima_updated_nobs
#                             arima_updated_nobs = len(in_sample_arima_y_diff)
                            
#                             if arima_logs:
#                                 st.markdown(f"<h1 style = 'text-align: center;'>Plot of differenced log {regressand_str}</h1>", unsafe_allow_html = True)
#                                 # Updating the sidebar to reflect on loaded data
#                                 index_1 += 1
#                                 st.sidebar.markdown(f"{index_1}. The **difference** of the **log {regressand_str}** has been taken. <br>The new number of observations is: **{arima_updated_nobs} (old: {arima_previous_nobs})**", unsafe_allow_html = True)                
                            
#                             else:
#                                 st.markdown(f"<h1 style = 'text-align: center;'>Plot of differenced {regressand_str}</h1>", unsafe_allow_html = True)
#                                 # Updating the sidebar to reflect on loaded data
#                                 index_1 += 1
#                                 st.sidebar.markdown(f"{index_1}. The **difference** of the **{regressand_str}** has been taken. <br>The new number of observations is: **{arima_updated_nobs} (old: {arima_previous_nobs})**", unsafe_allow_html = True)                
                            
#                             # Plotting exercise
#                             if arima_logs:   
#                                 label_name_2 = f"Differenced log {regressand_str}" # diffrencesof the log data
#                             else:
#                                 label_name_2 = f"Differenced {regressand_str}" # diffrencesof the log data
                            
#                             # Plotting exercise

#                             plotting_exercise(regressand = in_sample_arima_y_diff,
#                                       label_name = label_name_2,
#                                       dates_checkbox = dates_checkbox,
#                                       dates = dates)

#                             st.markdown("---")

#                             # Correlograms for differenced data
#                             # This is to see what number to submit as d
#                             if arima_logs and take_difference2:
#                                 st.markdown(f"<h1 style = 'text-align: center;'>Correlograms for differenced log {regressand_str}</h1>", unsafe_allow_html = True)
#                             elif arima_logs:
#                                 st.markdown(f"<h1 style = 'text-align: center;'>Correlograms for log {regressand_str}</h1>", unsafe_allow_html = True)
#                             elif take_difference2:
#                                 st.markdown(f"<h1 style = 'text-align: center;'>Correlograms for differenced {regressand_str}</h1>", unsafe_allow_html = True)
#                             else:
#                                 st.markdown(f"<h1 style = 'text-align: center;'>Correlograms for {regressand_str}</h1>", unsafe_allow_html = True)

#                             # Plotting ACF and PACF
#                             acf_pacf(series1 = in_sample_arima_y_diff, series2 =in_sample_arima_y_diff)

#                             # Conducting Augmented Dickey Fuller test on the differenced log data
#                             st.markdown("<h1 style = 'text-align: center;'>Dickey fuller test on differenced data</h1>", unsafe_allow_html = True)
#                             adf_test_2 = adfuller(in_sample_arima_y_diff)

#                             if adf_test_2[1] > 0.05:
#                                 st.write("")
#                                 st.write(f"p value is **{round(adf_test_2[1], 3)}**. Hence, There is not enough information to reject Ho. \n**We have evidence of a unit root**.")
#                                 st.write(f"**Data is not stationary**.")
#                             else:
#                                 st.write("")
#                                 st.write(f"p value is **{round(adf_test_2[1], 3)}**. Hence, we have enough evidence to reject Ho. \nThere is **no evidence of a unit root**.")
#                                 st.write(f"**Data is stationary**.")
#                         st.markdown("---")

#                         # Conducting a regression for the manual model

#                         st.markdown("<h1 style='text-align: center;'>Step 2: Estimation</h1>", unsafe_allow_html=True)

#                         # Breaking the screen into three columnds and asking to submit three numbers
#                         col1, col2, col3  = st.columns(3)
#                         one = col1.number_input("First num?:", step = 1, min_value = 0)
#                         two = col2.number_input("Second num?:", step = 1, min_value = 0)
#                         three = col3.number_input("Third num?:", step = 1, min_value = 0)

#                         # Fitting an ARIMA model
#                         if one == 0 and two == 0 and three == 0:
#                             st.write("Please, indicate your numbers for ARIMA guess")
#                             model_fit = None
                            
#                         elif (one is not None) and (two is not None) and (three is not None):
#                             model = conduct_manual_arima(in_sample_arima_y = in_sample_arima_y,
#                                                         one = one, 
#                                                         two = two, 
#                                                         three = three)

#                             model_fit = model.fit()
#                             st.write(model_fit.summary())   
#                             st.markdown("---")  

#                             # Updating the sidebar to reflect on loaded data
#                             index_1 += 1
#                             st.sidebar.markdown(f"{index_1}. **Manual** ARIMA model of order **{one, two, three}** has been estimated.")                
            
#                             # Retrieving fitted values and residuals
#                             fitted_values_manual_arima = model_fit.fittedvalues
#                             residuals_manual_arima = model_fit.resid

#                         else:
#                             st.write("Please, indicate your numbers for ARIMA guess")
#                             st.markdown("---")

#                         # Plotting the residuals and their density function
#                         if model_fit:
#                             st.markdown("<h1 style = 'text-align: center;'>Plotting the residuals and density</h1>", unsafe_allow_html = True)
#                             fig4, residuals = resid_kde(_model_fit = model_fit)
#                             plt.tight_layout()
#                             st.pyplot(fig4)
#                             st.markdown("---")

#                             # Checking residuals for the presence of autocorrelation
#                             # Errors (residuals) must be a white noise ie must not have any autocorrelation
#                             st.markdown("<h1 style = 'text-align: center;'>Correlograms for residuals</h1>", unsafe_allow_html = True)

#                             # Plot ACF and PACF
#                             acf_pacf(series1 = residuals_manual_arima, series2 = residuals_manual_arima)
#                             st.markdown("---")

#                             # Forecasting and plotting the forecast from the manual procedure
#                             st.markdown("<h1 style = 'text-align: center;'>Plotting manual forecasts</h1>", unsafe_allow_html = True)
#                             arima_manual_forecasts = model_fit.forecast(len(out_of_sample_arima_y))

#                             arima_regressand["arima_manual_forecasts"] = [None]*len(in_sample_arima_y) + list(arima_manual_forecasts)
                            
#                             # Plotting exercise
#                             if arima_logs:   
#                                 label_name_3 = f"Log {regressand_str}"
#                             else:
#                                 label_name_3 = f"{regressand_str}"

#                             # Plotting exercise
#                             plotting_exercise(regressand = arima_regressand,
#                                       label_name = label_name_3,
#                                       dates_checkbox = dates_checkbox,
#                                       dates = dates)

#                             st.markdown("---")

#                             # Calculating the loss functions for manual ARIMA model 
                            
#                             st.markdown("<h1 style = 'text-align: center;'>Statistical loss functions for manual ARIMA model</h1>", unsafe_allow_html = True)
#                             name2 = "Manual ARIMA model"
#                             theils_2 = theils_u(actual = out_of_sample_arima_y, forecast = arima_manual_forecasts)
#                             loss_functions_2 = loss_function(out_of_sample_data = out_of_sample_arima_y, forecasts = arima_manual_forecasts, column_name = name2, theil_number = theils_2)
#                             st.write(loss_functions_2)
#                             st.markdown("---")

#                             # Comparing all existing loss functions for all created forecasting models                         

#                             list_of_loss_functions = []
#                             if "loss_functions_1" in globals():
#                                 list_of_loss_functions.append(loss_functions_1)
#                             if "loss_functions_2" in globals():
#                                 list_of_loss_functions.append(loss_functions_2)
#                             #if "loss_functions_3" in globals():
#                                 #list_of_loss_functions.append(loss_functions_3)
#                             df = pd.DataFrame()

#                             if len(list_of_loss_functions) > 1:
#                                 if st.checkbox("Tick to compare the loss functions of all created forecasting models.", key = "1"):
#                                     st.markdown("<h1 style = 'text-align: center;'>Comparing all existing loss functions for all created forecasting models</h1>", unsafe_allow_html = True)
                        
#                                     for i in list_of_loss_functions:
#                                         df = pd.concat([df, i], axis = 1)
#                                     styled_df = df.style.apply(color_code, axis=1)                                  
#                                     st.markdown(styled_df.to_html(), unsafe_allow_html=True)
#                 st.markdown("---")              

# # Automatic ARIMA           
#     if y_var:   
#         if allow_analysis:
#             do_automatic_arima = st.checkbox("Tick to create an automatic ARIMA model.")
#             if do_automatic_arima:

#                 # Making a comment in the sidebar
#                 st.sidebar.markdown("---")
#                 st.sidebar.markdown("<h1 style='text-align: center;'>ARIMA  Automatic Analysis</h1>", unsafe_allow_html=True)
            
#                 # Retrieving the number of observations
#                 auto_arima_previous_nobs = len(auto_arima_regressand)

#                 # Plotting exercise
#                 label_name_5 = f"{regressand_str}"

#                 # Plotting exercise # this is the one that messes everything up
#                 st.markdown(f"<h1 style='text-align: center;'>Visual of {regressand_str}</h1>", unsafe_allow_html=True)

#                 plotting_exercise(regressand = auto_arima_regressand,
#                             label_name = label_name_5,
#                             dates_checkbox = dates_checkbox,
#                             dates = dates)                   

#                 st.markdown("---")

#                 # An option to take logs

#                 auto_arima_logs = st.checkbox(f"Tick to take **logs** of {regressand_str}.", key = "logs_3")
#                 if auto_arima_logs:
                    
#                     # Taking the actual logs
#                     auto_arima_regressand = np.log(auto_arima_regressand).copy()

#                     # Updating the sidebar to reflect on loaded data
#                     index_1 += 1
#                     st.sidebar.markdown(f"{index_1}. The **logs** of {regressand_str} have been taken.")

#                     # Ensuring the None values are gone
#                     has_missing_4 = auto_arima_regressand.isnull().values.any() # returns true of false
#                     if has_missing_4:
#                         auto_arima_regressand = auto_arima_regressand.dropna()

#                         # Resetting the index of the arima_regressand and getting the last index of the arima_regressand
#                         auto_arima_regressand = auto_arima_regressand.reset_index(drop=True)
                        
#                         # Extracting the new number of observations IF there are Nones
#                         auto_arima_updated_nobs = len(auto_arima_regressand)

#                         # Making a comment in the sidebar
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. After taking logs, **{regressand_str}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{auto_arima_updated_nobs} (old: {auto_arima_previous_nobs})**.", unsafe_allow_html=True)
#                     else:
#                         # Making a comment in the sidebar
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. After taking logs, **{regressand_str}** had no None values.<br>The number of observations is still: **{auto_arima_previous_nobs})**.", unsafe_allow_html=True)

#                         # Extracting the new number of observations IF there are Nones
#                         auto_arima_updated_nobs = len(auto_arima_regressand)

#                     st.markdown(f"<h1 style = 'text-align: center;'>Log {regressand_str}</h1>", unsafe_allow_html = True)    
                    
#                     # Plotting exercise
#                     label_name_6 = f"Log {regressand_str}"
#                     plotting_exercise(regressand = auto_arima_regressand,
#                                 label_name = label_name_6,
#                                 dates_checkbox = dates_checkbox,
#                                 dates = dates)     

#                 st.markdown("---")
        
#                 # No need to take differences because, the automatic ARIMA will choose this degree itself
#                 # Breaking the data into in sample and out of sample periods
#                 third_forecast = st.checkbox("Tick to perform in sample and out of sample forecating ability of the model.", key = "auto arima split 1")
#                 if third_forecast:
#                     st.write(f"**Please, note that this make take several minutes to complete.**")

#                     auto_arima_in_sample_proportion = st.number_input("Proportion for in-sample data", step = 0.05, value = 0.7, key = "auto arima split 2")
                
#                     if auto_arima_in_sample_proportion:
#                         if not "auto_arima_updated_nobs" in globals():
#                             auto_arima_updated_nobs = auto_arima_previous_nobs
                        
#                         out_of_sample_proportion_3 = 1 - auto_arima_in_sample_proportion
#                         auto_arima_in_sample_nobs = round(auto_arima_updated_nobs * auto_arima_in_sample_proportion) 

#                         # Updating the sidebar to reflect on loaded data
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. The in sample proportion is selected at **{round(auto_arima_in_sample_proportion * 100)}%**. Out of sample proportion is selected at **{round(100 - auto_arima_in_sample_proportion * 100)}%**.", unsafe_allow_html=True) 
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. The in sample number of observations is: **{auto_arima_in_sample_nobs} (before data splitting: {auto_arima_updated_nobs})**. The number of observations to forecast is: **{auto_arima_updated_nobs - auto_arima_in_sample_nobs}**.", unsafe_allow_html=True) 

#                         # Splitting the data into in sample and out of sample data
#                         in_sample_auto_arima_y, out_of_sample_auto_arima_y = auto_arima_regressand.iloc[:auto_arima_in_sample_nobs].copy(), auto_arima_regressand.iloc[auto_arima_in_sample_nobs:].copy()


#                         st.markdown("<h1 style = 'text-align: center;'>Automatic ARIMA regression results</h1>", unsafe_allow_html = True)    
#                         auto_arima = conduct_auto_arima(in_sample_auto_arima_y)

#                         st.write(f"Automatic ARIMA model suggests that the model is: {auto_arima}")
#                         st.write(auto_arima.summary())

#                         # Updating the activity log to say about the auto arima
#                         index_1 += 1
#                         st.sidebar.markdown(f"{index_1}. Automatic ARIMA model suggests that the model is: **{auto_arima.order}**.")
#                         st.markdown("---")

#                         # Generating and plotting the results
#                         st.markdown("<h1 style = 'text-align: center;'>Plotting automatic forecasts</h1>", unsafe_allow_html = True)    
#                         arima_auto_forecasts = auto_arima.predict(n_periods = len(out_of_sample_auto_arima_y))
#                         auto_arima_regressand["arima_auto_forecasts"] = [None] * len(in_sample_auto_arima_y) + list(arima_auto_forecasts)
                
#                         # Plotting exercise
#                         if auto_arima_logs:   
#                             label_name_7 = f"Log {regressand_str}"
#                         else:
#                             label_name_7 = f"{regressand_str}"

#                         # Plotting exercise
#                         plotting_exercise(regressand = auto_arima_regressand,
#                                             label_name = label_name_7,
#                                             dates_checkbox = dates_checkbox,
#                                             dates = dates)

#                     st.markdown("---")

#                     # Calculating the loss functions for manual ARIMA model                         
#                     st.markdown("<h1 style = 'text-align: center;'>Statistical loss functions for manual ARIMA model</h1>", unsafe_allow_html = True)
#                     name2 = "Automatic ARIMA model"
#                     theils_2 = theils_u(actual = out_of_sample_auto_arima_y, forecast = arima_auto_forecasts)
#                     loss_functions_3 = loss_function(out_of_sample_data = out_of_sample_auto_arima_y, 
#                                                         forecasts = arima_auto_forecasts, 
#                                                         column_name = name2, 
#                                                         theil_number = theils_2)
#                     st.write(loss_functions_3)
#                     st.markdown("---")

#                     # Comparing all existing loss functions for all created forecasting models 

#                     list_of_loss_functions = []
#                     if "loss_functions_1" in globals():
#                         list_of_loss_functions.append(loss_functions_1)
#                     if "loss_functions_2" in globals():
#                         list_of_loss_functions.append(loss_functions_2)
#                     if "loss_functions_3" in globals():
#                         list_of_loss_functions.append(loss_functions_3)
#                     df = pd.DataFrame()

#                     if len(list_of_loss_functions) > 1:
#                         if st.checkbox("Tick to compare the loss functions of all created forecasting models.", key = "2"):
#                             st.markdown("<h1 style = 'text-align: center;'>Comparing all existing loss functions for all created forecasting models</h1>", unsafe_allow_html = True)
                
#                             for i in list_of_loss_functions:
#                                 df = pd.concat([df, i], axis = 1)
#                             styled_df = df.style.apply(color_code, axis=1)                                  
#                             st.markdown(styled_df.to_html(), unsafe_allow_html=True)
#                             st.markdown("---")
