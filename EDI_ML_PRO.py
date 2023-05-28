import pandas as pd
import numpy as np
import pyttsx3 as py
from matplotlib import pyplot as p
import matplotlib 
%matplotlib inline
matplotlib.rcParams["figure.figsize"] = (20,10)
d1 = pd.read_csv("C:\\Users\\UBED\\EDI\\Bengaluru_House_Data.csv")
print("                                   This is the dataset                            ")
print()
py.speak("This is the Dataset")
print(d1.head())
print()
print("--------------------------------------------------------------------------")
print(f"shape = {d1.shape}")
py.speak(f'Total number of rows  = {d1.shape[0]} and  total number of  columns  = {d1.shape[1]}.')
print("--------------------------------------------------------------------------")
print()
print("These are the total counts of indivisul category  of area type only  :-")
print()
py.speak("These are the total counts of indivisual category  of area type only")
p = d1.groupby('area_type')['area_type'].agg('count')
print(p)
print()
print("--------------------------------------------------------------------------")
print()

'''Data Cleansing Here....... '''

print("Data cleansed here using certain techniques .....")
py.speak("Data cleansed here using certain techniques")
print()
d2 = d1.drop(['area_type','society','balcony','availability'],axis = 'columns')
print(d2)
print()
print()
print(d2.isnull().sum())
print()
print()
d3 = d2.dropna()
print(d3.isnull().sum())
print()
print("------------------------------------------------------------------------")
print(f"Now the shape is :- {d3.shape}")
print("------------------------------------------------------------------------")
py.speak(f"Now after some cleaning the data number of rows = {d3.shape[0]} and number of columns = {d3.shape[1]}")
print()
print(d3['size'].unique())
print()
print("------------------------------------------------------------------------")
  
''''Adding new 'BHK' column into dataframe to remove above arrors in size Column..... '''

d3['BHK'] = d3['size'].apply(lambda x : int(x.split(" ")[0]))
print()
print(d3.head())
print()
print('------------------------------------BHK Column Unique--------------------')
print()
print(d3['BHK'].unique())
print()

'''Rechecking of Columns .......'''

print()
print("-------------------------Errors found in 'Total_sqft' Column --------------")
print()
print(d3['total_sqft'].unique())
print()
print("-------------------------------------------------------------------------")

''' Got an error into 'Total_sqft' Column containing range values 
    which are INVALID ........ ''' 

''' So Let's apply some function to that Column in order to vanish 
   that errors and Cleansed them..... '''

def is_float(x):
    take = x.split('-')
    try:
        float(x)
    except:
        return False
    return True

print(d3[~d3['total_sqft'].apply(is_float)])
print()

''' Got the errors.....'''

    
def convert_sqft_to_num(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2 
    try :
        return float(x)
    except:
        return None 
    
d4 = d3.copy()   

d4['total_sqft'] = d4['total_sqft'].apply(convert_sqft_to_num)
d4.head()

print(d4.loc[30])
print()
print(d4.head(5))
print()
print("-----------------------Errors removed from 'Total_sqft'----------------------------------")
d5 = d4.copy()

'''It's very important to know the price per square feet (in lakhs)  so
  let's add one more column (price_per_sqft) to dataframe '''
  
print()
print()

d5['price_per_sqft'] = d5['price']*100000 / d5['total_sqft']

print(d5.head())
print()
print("---------------------------------------------------------------------------------------------")
print()

''' Grouping some categories of location i.e total no. of  
 flats in perticular area  '''
  
d5.location = d5.location.apply(lambda x : x.strip() )
location_stats = d5.groupby('location')['location'].agg('count').sort_values(ascending = False)
print(location_stats)
print()

''' Sorting the no. of flats accoring to areal quantity '''

d5.location = d5.location.apply(lambda x : 'other' if x in location_stats[location_stats<=10] else x)
print()
print()
print("                                  The Cleaned data is              ")
py.speak("The Cleaned data is ")
print()
''''print(len(d5.location.unique()))'''
print()
print(d5.head(10))
print()

print("-------------------------------------------------------------------------------------------")

''' Outlier Detection amd removal '''
print()
d6 = d5[~(d5.total_sqft/d5.BHK<300)]
print(d6.shape)
print()
print(d6.head())
print()
print(d6.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key ,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
        
    return df_out 

print()

d7 = remove_pps_outliers(d6)
print(d7.shape) 
print()
'''
def plot_scatter_chart(df,location):
    BHK2 = df[(df.location == location) & (df.BHK==2)]
    BHK3 = df[(df.location == location) & (df.BHK== 3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    b1 = BHK2.total_sqft,BHK2.price
    b2 = BHK3.total_sqft,BHK3.price
    p.scatter(b1,color = 'blue',label = '2 BHK', s = 50)
    p.scatter(b2,marker = "+" ,color = 'green',label = '3 BHK', s = 50)
    p.xlabel("Total Square Feet Area")
    p.ylabel("Price Per Square Feet")
    p.title(location)
    p.legend()
    

plot_scatter_chart(d7, 'Rajaji Nagar')
'''
def remove_bhk_outliers(df):
    e_i = np.array([])
    for location,location_df in df.groupby('location'):
        BHK_stats = { }
        for BHK , BHK_df in location_df.groupby('BHK'):
            BHK_stats[BHK] = {
                'mean' : np.mean(BHK_df.price_per_sqft),
                'std'  : np.std(BHK_df.price_per_sqft),
                'count': BHK_df.shape[0] 
                
            }
            
        for BHK , BHK_df in location_df.groupby('BHK'):
            stats = BHK_stats.get(BHK-1)
            if stats and stats['count'] > 5:
                e_i  = np.append(e_i, BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return  df.drop(e_i,axis ="index")

print()

d8  = remove_bhk_outliers(d7)
print("---------------------------------------------------------------------------------")
print(f'Now the shapeis  :-  {d8.shape}')
print("----------------------------------------------------------------------------------")      
py.speak(f"after removing outliers the number of rowes are {d8.shape[0]} and number of columns are{d8.shape[1]}")

'''plot_scatter_chart(d8,'RajajiNagar')
'''
print()
'''
import  matplotlib as plt
matplotlib.rcParams['figure.figsize'] = (20,10)
p.hist(d8.price_per_sqft,rwidth =0.8)
p.xlabel("Price Per Square Feet")
p.ylabel('count')
p.show()
'''
print(d8[d8.bath>10])
'''
plt.hist(d8.bath,rwidth = 0.8)
p.xlabel("Number o Bathrooms")
p.ylabel('count')
'''
print()
d9  = d8[d8.bath<d8.BHK+2]
print(d9.shape)
print()
print("--------------------------Final Dataset Is----------------------------------------")
d10 = d9.drop(['size','price_per_sqft'],axis ='columns')
print()
print(d10.head(3))
print()

''' Now lets do some Hot-encoding through 
    Pandas dummies module 
''' 

dummies = pd.get_dummies(d10.location)
print(dummies.head(3))
print()

d11 = pd.concat([d10,dummies.drop('other',axis='columns')],axis = 'columns')
print(d11.head())
print()
print()

d12  = d11.drop('location',axis = 'columns')
print(d12.head(3))
print('----------------------------------------------------------------------------------')
print(f'Now shape is :- {d12.shape}')
print('----------------------------------------------------------------------------------')
'''
let's go for model building

'''
print()
X = d12.drop('price',axis = 'columns')
print(X.head()) 

print()

Y = d12.price
print(Y.head())
print()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test   = train_test_split(X, Y,test_size = 0.2,random_state = 10)

from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()
lr_clf.fit(X_train,Y_train)

print(lr_clf.score(X_test ,Y_test))
print()


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5,test_size = 0.2 ,random_state=0)

print(cross_val_score(LinearRegression(), X,Y,cv=cv))


from sklearn.model_selection import GridSearchCV
#from sklearn.lasso import Lasso
from sklearn.tree import DecisionTreeRegressor
'''
          'lasso' : {
            'model' : Lasso(),
            'params': { 'alpha':[1,2] , 'selection':['random','cyclic'] }
            } ,
'''    
def find_best_model_using_grigsearchcv(X,Y):
    algos = {
        'linear_regression' : {
          'model' : LinearRegression(),
          'params' : { 'normalize' : [True , False] }
            },
        
          
        
        'decision_tree' :{
            
            'model' : DecisionTreeRegressor(),
            'params' : { 'criterion' : ['mse','friedman_mse'] , 'splitter' : ['best','random'] }
            
            }
        
         }

    scores = []
    cv = ShuffleSplit(n_splits=5,test_size = 0.2 ,random_state=0)
    for algo_name , config in algos.items():
        gs = GridSearchCV(config['model'], config['params'],cv=cv,return_train_score=False)
        gs.fit(X,Y)
        scores.append({ 'model': algo_name , 'best_score': gs.best_score_ , 'best_params': gs.best_params_} )
        
        
                      
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])    
        
print()   
print(find_best_model_using_grigsearchcv(X, Y))
print()
print()
print(X.columns)
print()
        
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
       x[loc_index] = 1
      
    return lr_clf.predict([x])[0]

print(predict_price('1st Phase JP Nagar', 1000, 2, 2),predict_price('1st Phase JP Nagar', 1000, 3, 3),end = "     " )
print()
import pickle
with open('banglore_home_price_model.pickle','wb') as f:
     pickle.dump(lr_clf,f)
        
     
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
    }
with open('columns.json','w') as f:
    f.write(json.dumps(columns))
        