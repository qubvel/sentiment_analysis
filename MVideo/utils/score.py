import sklearn
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def get_score(model, x, y, plot=True, sparse=50):
    
    y_pred = model.predict(x)  
    y_pred = np.clip(y_pred, 1.0, 5.0)
    
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    medae = median_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print ('{:.4} \nMSE: {:.4}\nMAE: {:.4}\nMedianAE: {:.4}\nR2 score: {:.4}'.format(model.name, mse, mae, medae, r2))
    
    if plot:
        plt.figure(figsize=(20,5))
        plt.title(model.name)
        plt.ylabel('Score')
        plt.plot(y_pred[::sparse])
        plt.plot(y[::sparse])
        plt.legend(('y_pred', 'y_test'))
        plt.show()
    
    return {'mean squared error':mse, 'mean absolute error':mae, 'median absolute error':medae, 'r2 score':r2}