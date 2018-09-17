"""Transformers

This module contains transformers for preprocessing data. Most operate on DataFrames and are named appropriately.
"""
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------
#from math import sqrt
#from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn.linear_model import LinearRegression, LogisticRegression

#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#from sklearn.metrics import mean_squared_error

from healthcareai.common.healthcareai_error import HealthcareAIError
SUPPORTED_IMPUTE_STRATEGY = ['MeanMedian', 'RandomForest']
#-----------------------------------------------------

###############################################################################

class DataFrameImputer( TransformerMixin ):
    
    """
    Impute missing values in a dataframe.

    Columns of dtype object or category (assumed categorical) are imputed with the mode (most frequent value in column).

    Columns of other types (assumed continuous) are imputed with mean of column.
    """
    def __init__(self, excluded_columns=None, impute=True, verbose=True, imputeStrategy='MeanMedian' ):
        self.impute = impute
        self.object_columns = None
        self.fill = None
        self.verbose = verbose
        
        self.impute_Object = None
        self.excluded_columns = excluded_columns
        self.imputeStrategy = imputeStrategy
        

    def fit(self, X, y=None):
        if self.impute is False:
            return self
        
        if ( self.imputeStrategy=='MeanMedian' or self.imputeStrategy==None ):
            # Grab list of object column names before doing imputation
            self.object_columns = X.select_dtypes(include=['object']).columns.values

            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O')
                                      or pd.api.types.is_categorical_dtype(X[c])
                                   else X[c].mean() for c in X], index=X.columns)

            if self.verbose:
                num_nans = sum(X.select_dtypes(include=[np.number]).isnull().sum())
                num_total = sum(X.select_dtypes(include=[np.number]).count())
                percentage_imputed = num_nans / num_total * 100
                print("Percentage Imputed: %.2f%%" % percentage_imputed)
                print("Note: Impute will always happen on prediction dataframe, otherwise rows are dropped, and will lead "
                      "to missing predictions")

            # return self for scikit compatibility
            return self
        elif ( self.imputeStrategy=='RandomForest' ):
            self.impute_Object = DataFrameImputerRandomForest( excluded_columns=self.excluded_columns )
            self.impute_Object.fit(X)
            return self
        elif ( self.imputeStrategy=='BaggedTree' ):
            self.impute_Object = DataFrameImputerBaggedTree()
            self.impute_Object.fit(X)
            return self
        else:
            raise HealthcareAIError('A imputeStrategy must be one of these types: {}'.format(SUPPORTED_IMPUTE_STRATEGY))

            

    def transform(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return X
        
        if ( self.imputeStrategy=='MeanMedian' or self.imputeStrategy==None ):
            result = X.fillna(self.fill)

            for i in self.object_columns:
                if result[i].dtype not in ['object', 'category']:
                    result[i] = result[i].astype('object')

            return result
        elif ( self.imputeStrategy=='RandomForest' ):
            result = self.impute_Object.transform(X)
            return result
        elif ( self.imputeStrategy=='BaggedTree' ):
            result = self.impute_Object.transform(X)
            return result
        else:
            raise HealthcareAIError('A imputeStrategy must be one of these types: {}'.format(SUPPORTED_IMPUTE_STRATEGY))




class DataFrameImputerBaggedTree(TransformerMixin):
    """
    Impute missing values in a dataframe.

    """
    def __init__(self, impute=True, verbose=True):
        self.impute = impute
        self.object_columns = None
        self.fill = None
        self.verbose = verbose

    def fit(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return self

        # Grab list of object column names before doing imputation
        self.object_columns = X.select_dtypes(include=['object']).columns.values

        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O')
                                  or pd.api.types.is_categorical_dtype(X[c])
                               else X[c].mean() for c in X], index=X.columns)

        if self.verbose:
            num_nans = sum(X.select_dtypes(include=[np.number]).isnull().sum())
            num_total = sum(X.select_dtypes(include=[np.number]).count())
            percentage_imputed = num_nans / num_total * 100
            print("Percentage Imputed: %.2f%%" % percentage_imputed)
            print("Note: Impute will always happen on prediction dataframe, otherwise rows are dropped, and will lead "
                  "to missing predictions")

        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return X

        result = X.fillna(self.fill)

        for i in self.object_columns:
            if result[i].dtype not in ['object', 'category']:
                result[i] = result[i].astype('object')

        return result
    


class DataFrameImputerRandomForest( TransformerMixin ):
    """
    Impute missing values in a dataframe.

    Columns of dtype object or category (assumed categorical) are imputed with the mode (most frequent value in column).

    Columns of other types (assumed continuous) are imputed with mean of column.
    """

    def __init__(self, excluded_columns=None, impute=True, verbose=True):
        self.impute = impute
        self.object_columns = None
        self.fill = None
        self.verbose = verbose

        self.excluded_columns = excluded_columns
        self.fill_dict = {}

    def fit(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return self
        
        #Replacing None by NaN, if any None is present in X
        X.fillna( value=np.nan, inplace=True)
        
        # Creating base copy of original Dataframe as 'main_df'
        main_df = X.copy()
        X_backup = X.copy()
        
        # Removing excluded cols
        all_columns = X.columns
        if( self.excluded_columns!=None ):
            for col in self.excluded_columns:
                if( col in all_columns ):
                    X.drop( columns= [col], axis=1, inplace=True )
        
        # Seperating all columns into Categorical and Numeric
        cat_list=[]
        num_list=[]
        for c in X:
            if( X[c].dtype == np.dtype('O') or pd.api.types.is_categorical_dtype(X[c]) ):
                cat_list.append( c )
            else:
                num_list.append( c )
            
        #--------------------------------------------------------------------------------------------------------------------------
        X_NumericImputed = self.getNumericImputedData( main_df=main_df.copy(), X=X.copy(), cat_list=cat_list, num_list=num_list )
        
        main_df_NumericImputed = X_NumericImputed.join( main_df[ cat_list ], how='outer').copy()
        X_backup = main_df_NumericImputed.copy()
        X = main_df_NumericImputed.copy()
        
        X_CategoricImputed = self.getCategoricalImputedData( main_df=main_df.copy(), X_NumericImputed=X_NumericImputed.copy(), X=X, cat_list=cat_list, num_list=num_list )
        
        X = main_df.copy()
        #--------------------------------------------------------------------------------------------------------------------------
        
        if self.verbose:
            num_nans = sum(X.select_dtypes(include=[np.number]).isnull().sum())
            num_total = sum(X.select_dtypes(include=[np.number]).count())
            percentage_imputed = num_nans / num_total * 100
            print("Percentage Imputed: %.2f%%" % percentage_imputed)
            print("Note: Impute will always happen on prediction dataframe, otherwise rows are dropped, and will lead "
                  "to missing predictions")
        
            print("------------------<< Fill-dictionary-Details >> ---------------------")
            self.printDictReport()

        # return self for scikit compatibility
        return self
    
    
    def transform(self, X, y=None):                
        # Return if not imputing
        if self.impute is False:
            return X
        
        #Replacing None by NaN, if any None is present in X
        X.fillna( value=np.nan, inplace=True)

        for colName, imputeData in self.fill_dict.items():
            if( colName in X.columns ):
                X.loc[ X[ colName ].isnull(), colName ] = imputeData        
        
        return X
   
    def printDictReport(self):
        sum=0
        print("***************** << fill_dict deatils >> *********************")
        for col, imputeData in self.fill_dict.items():
            sum+=len(imputeData)
            print( "colName                   = ", col )
            print( "total no of values filled = ", len(imputeData))
            print( "top 10 filled values      = ", imputeData[0:10])
        print("total no of values to be imputed in the dataframe = ", sum)
    
    
    def getNumericImputedData( self, main_df, X, cat_list, num_list ):
        X_backup = X
        to_impute = []
        all_columns = []
        predictor_columns = []
        for i in num_list:
            X = X_backup
            to_impute = [i]

            # Check if to_impute col have any NaN Values. If no NaN values, no need to do imputation for this column
            if ( X[ to_impute ].isnull().values.any()==False):
                #print("No Nul values in = {} column, skipping the imputation loop".format( [i] ) )
                continue

            all_columns = list(X.columns)
            predictor_columns = all_columns.copy()
            predictor_columns.remove( to_impute[0] )

            X = X[ predictor_columns ]

            # DataFrameImputer --> Impute the missing values in X (using Mean and Median)
            # Note: After every iteration we will have 1 col less, in list of Cols having NaN values
            X = self.getTempImutedData( X )

            # As we didnt imputed mising values of to_impute col (since they are to be imputed using ML)
            # Now joining to_impute col(having NaN values back to X)
            X = X.join( main_df[ to_impute ], how='outer')

            # DataFrameCreateDummyVariables
            # Converting Categorical COls --> to Numeric so that they can be feeded to ML model
            columns_to_dummify = list(X.select_dtypes(include=[object, 'category']))
            X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

            # Since some new cols are created after get_dummies, updating the predictor_columns List
            predictor_columns = list(X.columns)
            predictor_columns.remove(to_impute[0])
            
            # Get the predicted values for missing data
            y_pred_main = self.getImputePredictions(X=X, predictor_columns=predictor_columns, to_impute=to_impute, toImputeType='numeric' )
            
            # add the imputation data to the fill_dict
            self.fill_dict[ to_impute[0] ] = y_pred_main

            # updating the imputed NaN of to_impute col in, X_backup
            # Now in next iteration this X_backup will be used as base DF
            X_backup.loc[ X_backup[ to_impute[0] ].isnull(), to_impute[0] ] = self.fill_dict[ to_impute[0] ]
            
        X_NumericImputed = X_backup[ num_list ].copy()
        return X_NumericImputed


    def getCategoricalImputedData( self, main_df, X_NumericImputed, X, cat_list, num_list ):
        X_backup = X
        to_impute = []
        all_columns = []
        predictor_columns = []
        for i in cat_list:
            X = X_backup
            to_impute = [i]

            # Check if to_impute col have any NaN Values. If no NaN values, no need to do imputation for this column
            if ( X[ to_impute ].isnull().values.any()==False):
                #print("No Nul values in = {} column, skipping the imputation loop".format( [i] ) )
                continue

            all_columns = list(X.columns)
            predictor_columns = all_columns.copy()
            predictor_columns.remove( to_impute[0] )

            # tempImpute_columns = List of cols to be imputed temporarily using median i.e ( cat_list - to_impute[0] ) 
            #** X_temp = X[ predictor_columns ]
            tempImpute_columns = cat_list.copy()
            tempImpute_columns.remove( to_impute[0] )

            X = X[ tempImpute_columns ]

            # DataFrameImputer --> Impute the missing values in X (using Mean and Median)
            # Note: After every iteration we will have 1 col less, in list of Cols having NaN values
            X = self.getTempImutedData( X )
            
            # X = X(tempImpute_columns) + X_NumericImputed + main_df(to_impute)
            # As we didnt imputed mising values of to_impute col (since they are to be imputed using ML)
            # Now joining to_impute col(having NaN values back to X)
            X = X.join( X_NumericImputed, how='outer')
            X = X.join( main_df[ to_impute ], how='outer')

            # DataFrameCreateDummyVariables
            # Converting Categorical COls --> to Numeric so that they can be feeded to ML model
            # columns_to_dummify = columns_to_dummify - to_impute[0], as to_impute is a categorcal but it is to be imputed using ML model
            columns_to_dummify = list(X.select_dtypes(include=[object, 'category']))
            if( to_impute[0] in columns_to_dummify ):
                columns_to_dummify.remove( to_impute[0] )
            else:
                print( "Col to_impute = {} not found in columns_to_dummify = {}, Raise exception".format(to_impute[0], str(columns_to_dummify) ) )
                raise HealthcareAIError( "Col to_impute = {} not found in columns_to_dummify = {}".format( to_impute[0], str(columns_to_dummify) ) )

                
            X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

            # Since some new cols are created after get_dummies, updating the predictor_columns List
            predictor_columns = list(X.columns)
            predictor_columns.remove(to_impute[0])

            # Sice target col i.e to_impute[0] is a categorical feature, we have to convert it into indexed format(i.e 0,1,2)
            target_column = to_impute[0]
            from_List = list( X[target_column].unique() )

            if np.NaN in from_List:
                from_List.remove(np.NaN)
            elif (np.isnan( from_List ).any() ):
                #from_List.remove( np.NaN )
                from_List = [i for i in from_List if str(i) != 'nan']
            else:
                print("Null values didn't captured properly for col = {}, having unique values as = {} raise exception".format( to_impute[0], from_List )  )
                raise HealthcareAIError( "Null values didn't captured properly for col = {}, having unique values as = {}".format( to_impute[0], from_List ) )
                
                
            #check if there is only and only 1 datatype, else rasie exception
            from_List.sort()
            to_List = [ i for i in range( 0,len(from_List) ) ]
            X[ target_column] = X[ target_column ].replace( from_List, to_List, inplace=False)
            
            # Get the predicted values for missing data
            y_pred_main = self.getImputePredictions( X=X, predictor_columns=predictor_columns, to_impute=to_impute, toImputeType='categorical' )
            
            
            # updating the imputed NaN of to_impute col in, X_backup
            # Now in next iteration this X_backup will be used as base DF
            X_backup.loc[ X_backup[ to_impute[0] ].isnull(), to_impute[0] ] = y_pred_main

            # Reconverting the idexed-to_impute column into its original form
            from_List, to_List = to_List, from_List
            X_backup[ to_impute] = X_backup[ to_impute ].replace( from_List, to_List, inplace=False)
            
            # add the imputation data to the fill_dict    
            y_pred_main_df = pd.DataFrame( data=y_pred_main, columns=to_impute )
            y_pred_main_df[ to_impute] = y_pred_main_df[ to_impute ].replace( from_List, to_List, inplace=False)            
            self.fill_dict[ to_impute[0] ] = y_pred_main_df[ to_impute[0] ].values
            
        X_CategoricImputed = X_backup[ cat_list ].copy()
        return X_CategoricImputed


    def getTempImutedData( self, X ):
        object_columns = X.select_dtypes(include=['object']).columns.values
        fill = pd.Series( [ X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') or pd.api.types.is_categorical_dtype(X[c])
                            else X[c].mean() 
                            for c in X
                          ]

                          , index=X.columns)
        result = X.fillna( fill )
        for i in object_columns:
                    if result[i].dtype not in ['object', 'category']:
                        result[i] = result[i].astype('object')
      
        return result.copy()
    
    
    def getImputePredictions( self, X, predictor_columns, to_impute, toImputeType ):
        # Seperating the mainDf into train(dont have NaN) and test(having NaN) data
        train = X[ X[to_impute[0]].isnull()==False ]
        test  = X[ X[to_impute[0]].isnull() ]

        # General X, y used for train test split
        # X_main = DF based on which we have to predict the NaN of to_impute col
        X = train[ predictor_columns ]
        #y = train[ to_impute ]      # now trying to use y as a array not Df, to avoid warning
        y = train[ to_impute[0] ].values
        X_main = test[ predictor_columns ]

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100 )

        if( toImputeType=='numeric' ):
            algo = RandomForestRegressor( random_state=100 )
        elif( toImputeType=='categorical' ):
            algo = RandomForestClassifier( random_state=100 )
        else:
            print("Raise action, as invalid predictorType")
                
        fit_algo = algo.fit( X_train, y_train )
        y_pred = fit_algo.predict( X_test )
        y_pred_main = fit_algo.predict( X_main )
        return y_pred_main.copy()


######################################################################################################################################


class DataFrameConvertTargetToBinary(TransformerMixin):
    # TODO Note that this makes healthcareai only handle N/Y in pred column
    """
    Convert classification model's predicted col to 0/1 (otherwise won't work with GridSearchCV). Passes through data
    for regression models unchanged. This is to simplify the data pipeline logic. (Though that may be a more appropriate
    place for the logic...)

    Note that this makes healthcareai only handle N/Y in pred column
    """

    def __init__(self, model_type, target_column):
        self.model_type = model_type
        self.target_column = target_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO: put try/catch here when type = class and predictor is numeric
        # TODO this makes healthcareai only handle N/Y in pred column
        if self.model_type == 'classification':
            # Turn off warning around replace
            pd.options.mode.chained_assignment = None  # default='warn'
            # Replace 'Y'/'N' with 1/0
            X[self.target_column].replace(['Y', 'N'], [1, 0], inplace=True)

        return X


class DataFrameCreateDummyVariables(TransformerMixin):
    """Convert all categorical columns into dummy/indicator variables. Exclude given columns."""

    def __init__(self, excluded_columns=None):
        self.excluded_columns = excluded_columns

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        columns_to_dummify = list(X.select_dtypes(include=[object, 'category']))

        # remove excluded columns (if they are still in the list)
        for column in columns_to_dummify:
            if column in self.excluded_columns:
                columns_to_dummify.remove(column)

        # Create dummy variables
        X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

        return X


class DataFrameConvertColumnToNumeric(TransformerMixin):
    """Convert a column into numeric variables."""

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        X[self.column_name] = pd.to_numeric(arg=X[self.column_name], errors='raise')

        return X


class DataFrameUnderSampling(TransformerMixin):
    """
    Performs undersampling on a dataframe.
    
    Must be done BEFORE train/test split so that when we split the under/over sampled dataset.

    Must be done AFTER imputation, since under/over sampling will not work with missing values (imblearn requires target
     column to be converted to numerical values)
    """

    def __init__(self, predicted_column, random_seed=0):
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO how do we validate this happens before train/test split? Or do we need to? Can we implement it in the
        # TODO      simple trainer in the correct order and leave this to advanced users?

        # Extract predicted column
        y = np.squeeze(X[[self.predicted_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = X.drop([self.predicted_column], axis=1)

        # Initialize and fit the under sampler
        under_sampler = RandomUnderSampler(random_state=self.random_seed)
        x_under_sampled, y_under_sampled = under_sampler.fit_sample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_under_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_under_sampled = pd.Series(y_under_sampled)
        result[self.predicted_column] = y_under_sampled

        return result


class DataFrameOverSampling(TransformerMixin):
    """
    Performs oversampling on a dataframe.

    Must be done BEFORE train/test split so that when we split the under/over sampled dataset.

    Must be done AFTER imputation, since under/over sampling will not work with missing values (imblearn requires target
     column to be converted to numerical values)
    """

    def __init__(self, predicted_column, random_seed=0):
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO how do we validate this happens before train/test split? Or do we need to? Can we implement it in the
        # TODO      simple trainer in the correct order and leave this to advanced users?

        # Extract predicted column
        y = np.squeeze(X[[self.predicted_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = X.drop([self.predicted_column], axis=1)

        # Initialize and fit the under sampler
        over_sampler = RandomOverSampler(random_state=self.random_seed)
        x_over_sampled, y_over_sampled = over_sampler.fit_sample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_over_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_over_sampled = pd.Series(y_over_sampled)
        result[self.predicted_column] = y_over_sampled

        return result


class DataFrameDropNaN(TransformerMixin):
    """Remove NaN values. Columns that are NaN or None are removed."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Uses pandas.DataFrame.dropna function where axis=1 is column action, and
        # how='all' requires all the values to be NaN or None to be removed.
        return X.dropna(axis=1, how='all')


class DataFrameFeatureScaling(TransformerMixin):
    """Scales numeric features. Columns that are numerics are scaled, or otherwise specified."""

    def __init__(self, columns_to_scale=None, reuse=None):
        self.columns_to_scale = columns_to_scale
        self.reuse = reuse

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Check if it's reuse, if so, then use the reuse's DataFrameFeatureScaling
        if self.reuse:
            return self.reuse.fit_transform(X, y)

        # Check if we know what columns to scale, if not, then get all the numeric columns' names
        if not self.columns_to_scale:
            self.columns_to_scale = list(X.select_dtypes(include=[np.number]).columns)

        X[self.columns_to_scale] = StandardScaler().fit_transform(X[self.columns_to_scale])

        return X
