
import numpy as np
import pandas as pd 
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from fbprophet.diagnostics import cross_validation
from dateutil.relativedelta import *


def load_data(url):
        
    return pd.read_csv(url)

def mean_absolute_percentage_error(y_true, y_pred): 
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




df = load_data(r'C:/Users/azura/Desktop/AI Projects/Artificial Intelligence/Datasets/Open source datasets/shampoo_sales.csv')

df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)

latest = []
mean_values = [] 
true_values = []
pred_values = []

for i in range (1, 13):
    
    train = df.drop(df.index[23 + i:])  
    date = train['ds'].max() + relativedelta(months =+ 1)
    latest = [date]
    latest = pd.DataFrame(latest, columns = ['ds'])
    model = Prophet(yearly_seasonality = False)
    model.fit(train)
    forecast = model.predict(latest)
    y_true = df['y'].loc[df['ds'] == date].values
    y_pred = forecast['yhat'].values
    true_values.append(y_true)
    # true_values = pd.DataFrame(true_values, columns = ['y'])
    pred_values.append(y_pred)
    # pred_values = pd.DataFrame(pred_values, columns = ['y'])
    # print(y_true, y_pred)  
    
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) 
    #make a new list of all MAPE values for each iteration 
    mean_values.append(mape)
    apples = sum(mean_values)/len(mean_values)
    # print('MAE: %.3f' % mae)
    # print('MAPE: %.3f' % mape)  
    # plt.plot(y_true, label='Actual')
    # plt.plot(y_pred, label='Predicted')
    # plt.legend()
    # plt.show()
    # print(y_true, y_pred) 
    
    
    
 
print (apples) 
plt.plot(true_values, label='Actual')
plt.plot(pred_values, label='Predicted')
plt.legend()
plt.show()


# train = df.drop(df.index[-12:])
# model = Prophet(yearly_seasonality=True)
# # model = Prophet()
# model.fit(train)

# future = list()
# for i in range(1, 13):
#  	date = '2003-%02d' % i
#  	future.append([date])


# future = pd.DataFrame(future)
# future.columns = ['ds']
# future['ds']= pd.to_datetime(future['ds'])     

# forecast = model.predict(future)

# # calculate MAE between expected and predicted values for december
# y_true = df['y'][-12:].values
# y_pred = forecast['yhat'].values
# mae = mean_absolute_error(y_true, y_pred)
# mape = mean_absolute_percentage_error(y_true, y_pred) 
# print('MAE: %.3f' % mae)
# print('MAPE: %.3f' % mape)  



# plt.plot(y_true, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.legend()
# plt.show()

# model.plot_components(forecast)

# model = Prophet()
# model.fit(df)
# df_cv = cross_validation(model, initial='730 days', period='30 days', horizon = '14 days')
















































































