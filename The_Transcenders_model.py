# -*- coding: utf-8 -*-
# Imports
import pandas as pd          
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import pickle
seed = 98   # to provide a unique seed to get the same results

# Loading the datasets provided.
price = pd.read_csv('Hackathon_case_training_hist_data.csv')
dataset = pd.read_csv('Hackthon_case_training_data.csv')
train = pd.read_csv('Hackthon_case_training_output.csv')
id_test = pd.read_excel('sample_ouput.xlsx')

filename = 'finalized_model.sav'

# Function to extract the relevant features for predictive analysis
def feature_extraction(price,dataset,train,id_test):
    dataset.info()  # getting information about the dataset.
    missing_value_count(dataset)  #counts the missing values and plots the result.
    #The list of columns to be dropped having missing values > 50%.
    drop_columns = ['activity_new','campaign_disc_ele','date_first_activ','forecast_base_bill_year',
                    'forecast_base_bill_ele','forecast_cons','forecast_bill_12m']
    for col in drop_columns:
        dataset = dataset.drop([col],axis=1)   # columns irrelevant are dropped.

    '''price dataset contains different prices for 1-12 days.
    the median value of the id is taken with the help of group function.
    '''
    price_data1 = price.groupby(['id']).median() 
    price_corr = price_data1.corr() # for finding the correlation between the prices and dropping columns which are correlated.
    
    pri_df = pd.DataFrame()             # creating a new pri_df dataframe.
    # putting the desired values after correlation measure into pri_df.
    pri_df['id'] = price_data1.index
    pri_df['price_p1_var'] = price_data1['price_p1_var'].values
    pri_df['price_p2_var'] = price_data1['price_p2_var'].values
    pri_df['price_p1_fix'] = price_data1['price_p1_fix'].values
    # merging the dataset with the price dataset and storing it into a new dataframe for further processing.
    df = pd.merge(dataset, pri_df,how='inner', on='id')
    
    # Creating a different dataframe for visualization purpose.
    df_vis = pd.merge(df, pri_df,how = 'inner',on ='id')
    df_vis = pd.merge(dataset,train, how= 'inner',on = 'id')
    
    # Working with date columns
    date_col = ['date_activ','date_end','date_modif_prod','date_renewal' ]
    
    ''' The date columns has been converted into datetime to extract relevant 
    details like date, year, month etc.
    here we have extracted days and diffference is calculated.
    '''
    for date in date_col:
        df[date] = pd.to_datetime(df[date]) 
    dates_diff = pd.DataFrame()
    days = pd.DataFrame()
    dates_diff['date_end-activ'] = df['date_end']-df['date_activ']
    days['days_end-activ']=dates_diff.iloc[:,0].dt.days
    dates_diff['date_modif-_activ']=df['date_modif_prod']-df['date_activ']
    days['date_modif-activ'] = dates_diff.iloc[:,1].dt.days
    dates_diff['date_end-renewal']=df['date_end']-df['date_renewal']
    days['date_end-renewal'] = dates_diff.iloc[:,2].dt.days
   
    dates_diff['date_end-modif']=df['date_end']-df['date_modif_prod']
    days['date_end-modif'] = dates_diff.iloc[:,3].dt.days
   
    # Dropping the date columns after processing information from it and storing in df.
    for col in date_col:
        df = df.drop([col],axis=1)
    #concat function of pandas joins days dataframe with processing dataframe.
    df = pd.concat([df,days],axis=1)
    
    corr = df.corr() # finding correlation between columns of df.
    sns.heatmap(corr) #visualization of the heatmap
    # list of columns to be dropped after correlatiion.
    drop_corr_columns = ['cons_last_month','forecast_cons_12m','forecast_cons_year','forecast_price_energy_p1',
                         'forecast_price_energy_p2', 'forecast_price_pow_p1','days_end-activ','margin_gross_pow_ele']
    for col in drop_corr_columns:
        df = df.drop([col],axis=1)
    # filling the null values with zeroes.
    for col in df.columns:
        df[col] = df[col].fillna(0)
    # merging the dataframe with training set for model preparation.
    df1 = pd.merge(df,train,how = 'inner', on = 'id')
    prediction = pd.merge(df,id_test,how ='inner',on='id')
    # dropping the id axis for training and testing.
    df1 = df1.drop(['id'],axis=1)
    dummies= pd.get_dummies(df1) #one hot encoding the categorical columns
    
    id_pred = prediction['id'].values
    prediction = prediction.drop(['id'],axis=1)
    prediction = prediction.drop(['Probability'],axis=1)
    prediction = prediction.drop(['churn'],axis=1)
    pred_dummies = pd.get_dummies(pred_dummies)
    # removing the dummy variable trap
    drop_col = ['channel_sales_sddiedcslfslkckwlfkdpoeeailfpeds','channel_sales_0','has_gas_t',
            'channel_sales_fixdbufsefwooaasfcxdxadsiekoceaa','channel_sales_epumfxlbckeskwekxbiuasklxalciiuu',
            'origin_up_usapbepcfoloekilkwsdiboslwaxobdp','origin_up_ewxeelcelemmiwuafmddpobolfuxioce','origin_up_0']
    
    for col in drop_col:
         dummies= dummies.drop([col],axis=1)
     # for prediction set.   
    for col in drop_col:
         dummies= dummies.drop([col],axis=1)
    
    return dummies,df_vis,pred_dummies  #returning the visualization and dummies dataframe.

def model_performanence(data):
    # creating feature and target dataset.
    X = data.drop(['churn'],axis=1)
    y = data['churn']
    # splitting the dataset into train and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)
    # function to normalize the X dataframe.
    def normalize(df):
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    X =normalize(X) # Xis now narmalized between 0 and 1.
    
    # function for grid-search to chose beast parameters.
    def gridsearch(model, scorer, param_grid, X, y):
        '''     
        Perform Gridsearch for the given model and return the best model
        '''
        grid_search = GridSearchCV(model, param_grid=param_grid, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        print(grid_search.best_estimator_)
        print("Grid scores on development set:")
        return model

    # Dealing with the problem of imbalanced dataset by oversampling with smote.
    
    #Up-sampling by creating synthetic data rather than multiplying the minority data.
    sm = SMOTE(random_state=seed, ratio = 1.0)
    x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    # chosing best parameters for random forest for feature selection.
    rf = RandomForestClassifier()
    recall = make_scorer(recall_score)
    param_grid = dict(n_estimators=[20, 40, 100],
                  max_depth = [4,6,8],
                  max_features = ['sqrt', 'log2', None],
                  class_weight=[{1: w} for w in [1 , 2, 4,6]]
                    ,bootstrap = [True, False])
    rf = gridsearch(rf, recall, param_grid, x_train_res, y_train_res)
    rf.fit(x_train_res, y_train_res)
    
    # plotting the feature importance graph.
    features_label = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(X_train.shape[1]):
        print ("%2d) %-*s %f" % (i + 1, 30, features_label[i], importances[indices[i]]))

    plt.title('Random Forest Feature Importances ')
    plt.bar(range(X_train.shape[1]), importances[indices], color = "green", align = "center")
    plt.xticks(range(X_train.shape[1]), features_label, rotation = 90)
    plt.show()

    #### Model selection
    lrMod = LogisticRegression(penalty = 'l2', dual = False, tol = 0.0001, C = 1.0, fit_intercept = True,
                            intercept_scaling = 1, class_weight = None, 
                            random_state = None, solver = 'liblinear', max_iter = 100,
                            multi_class = 'ovr', verbose = 2)
    # Fitting the model with training data 
    lrMod.fit(x_train_res, y_train_res)
    # Compute the model accuracy on the given test data and labels
    lr_acc = lrMod.score(X_test, y_test)
    # Return probability estimates for the test data
    test_labels = lrMod.predict_proba(np.array(X_test))[:,1]

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    lr_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)


    # AdaBoost Classifier
    adaMod = AdaBoostClassifier(base_estimator = None, n_estimators = 1000, learning_rate = 1.0)
    # Fitting the model with training data 
    adaMod.fit(x_train_res, y_train_res)
    # Compute the model accuracy on the given test data and labels
    ada_acc = adaMod.score(X_test, y_test)
    # Return probability estimates for the test data
    test_labels = adaMod.predict_proba(np.array(X_test))[:,0]

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    ada_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

    # Initialization of the GradientBoosting model
    gbMod = GradientBoostingClassifier(loss = 'deviance', n_estimators = 200,)
    # Fitting the model with training data 
    gbMod.fit(x_train_res, y_train_res)
    # Compute the model accuracy on the given test data and labels
    gb_acc = gbMod.score(X_test, y_test)
    # Return probability estimates for the test data
    test_labels = gbMod.predict_proba(np.array(X_test))[:,1]

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    gb_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

    # Compute the model accuracy on the given test data and labels
    rf_acc = rf.score(X_test, y_test)
    # Return probability estimates for the test data
    test_labels = rf.predict_proba(np.array(X_test))[:,1]

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    rf_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

   # Extra treeClassifier model.
    extc = ExtraTreesClassifier(n_estimators=500,max_features= 20,criterion= 'gini',min_samples_split= 8,
                            max_depth= 70, min_samples_leaf= 2) 
    extc.fit(x_train_res,y_train_res)
    ext_acc = lrMod.score(X_test, y_test)
    # Return probability estimates for the test data
    test_labels = lrMod.predict_proba(np.array(X_test))[:,1]
    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    ext_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

    # plotting the ROC curve for better understanding
    classifiers = [lrMod, adaMod, gbMod,rf, extc]
    classifier_names = [ 'Logistic Regression', 'AdaBoost', 'GradientBoosting', 'Random Forest','Extra Tree Classifier']
    auc_scores, roc_plot = plot_roc_curve(classifiers, classifier_names, "ROC curve", X_test, y_test)
    roc_plot
    #10 fold cross validation of the estimators using StratifiedKFold.
    cv = StratifiedKFold(n_splits=10)
    for clf in classifiers:
        print (cross_val_score(clf, X, y, scoring='roc_auc', cv=cv).mean())
    
    
    # The extra tree classifer
    adaEtc = AdaBoostClassifier(base_estimator=extc,n_estimators = 100,learning_rate=0.1,random_state=seed)
    adaEtc.fit(x_train_res,y_train_res)
    #printing the classification report
    print(classification_report(adaEtc.predict(X_test),y_test))
    # save the model to disk
    
    pickle.dump(adaEtc, open(filename, 'wb'))
 
    
def visualization(data,df_vis):
    # scipy library is used for creating dendograms of 
    from scipy.cluster import hierarchy as hc
    X = np.random.rand(20, 10)
    names = data.columns
    inverse_correlation = 1 - abs(data.corr())
    fig = ff.create_dendrogram(inverse_correlation.values, orientation='left', labels=names, colorscale=colors, linkagefun=lambda x: hc.linkage(x, 'average'))
    fig['layout'].update(dict(
            title="Dendogram of clustering the features according to correlation",
            width=800, 
            height=600,
            margin=go.layout.Margin(l=180, r=50),
            xaxis=dict(
                    title='distance',
                    ),
                    yaxis=dict(
            title='features',
            automargin=True,
            ),
            ))
    plot(fig, filename='dendrogram_corr_clustering')
    # count plot of the fields
    sns.countplot(x='channel_sales', hue = 'churn',data = df_vis)
    sns.countplot(x='origin_up', hue = 'churn',data = df_vis)
    sns.countplot(x='has_gas', hue = 'churn',data = df_vis)
    # churn and non churn count.
    churn     = df_vis[df_vis["churn"] == 1]
    not_churn = df_vis[df_vis["churn"] == 0]
    #creating seperate category and numerical columns for visualization.
    target_col = ["churn"]
    cat_cols   = df_vis.nunique()[df_vis.nunique() < 12].keys().tolist()
    cat_cols   = [x for x in cat_cols if x not in target_col]
    num_cols   = [x for x in df_vis.columns if x not in cat_cols + target_col]
    # pie plot from 
    plot_pie(cat_cols[0])
    plot_pie(cat_cols[1])
    #histogram plot.
    histogram(cat_cols[0])
    histogram(num_cols[15])
    # box plot for outlier detection.
    new_df = df_vis[num_cols[1:20]]
    gen_boxplot(new_df)
    data = trace
    plot(data)

def predict(to_pred):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)
    
    y_pred = adaEtc.predict_proba(to_pred)
    #print y_pred

    pd.DataFrame({"id": id_pred, "PredictedProb": y_pred[:,1]}).to_csv('output.csv',index=False)
    
    
#extracting the relevant data
data,df_vis,to_pred =feature_extraction(price,dataset,train,id_test)
visualization(data,df_vis)   # calling visualization
predict(to_pred)          # calling predict function

output = pd.read_csv('output.csv')
output['churn'] = output['PredictedProb']>0.3
output['churn'].replace([True,False],[1,0],inplace = True)

to_be_churned = output[output['churn']==1]
to_not_be_churned = output[output['churn']==0]

row_churn = to_be_churned.shape[0]
row_non_churn = to_not_be_churned.shape[0]

prediction = [row_churn,row_non_churn]
plt.pie(prediction,colors=['red','green'],autopct = '%.1f%%')
plt.title('Predicted Churn')
pd.DataFrame(output).to_csv('output.csv',index=False)
    
