from typing import Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class DebuggerTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print("DummyTransformer")
        return X



class CabinLetterExtractor(BaseEstimator, TransformerMixin):


    def __init__(self, letter:Union["first", "last"], verbose:bool=False):

        self.__assert_letter(letter=letter)

        self.letter = letter
        self.verbose = verbose

    def __assert_letter(self, letter):

        assert_message = "Valid letters are `first` or `last`."
        assert_message += f"You have passed {letter}"

        assert letter in ["first", "last"], assert_message

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if self.verbose:
            print("CabinLetterExtractor")
            print(type(X))

        if self.letter == "first":
            return X.iloc[:, 0].apply(lambda cabin: cabin[0] if cabin != "NA" else cabin).to_frame()
        else:
            return X.iloc[:, 0].apply(lambda cabin: cabin[-1] if cabin != "NA" else cabin).to_frame()



class SurnameExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, verbose:bool=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.iloc[:, 0].apply(lambda name: name.split(" ")[1] if name != "NA" else name).to_frame()

    
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, list_surnames):
        self.list_surnames = list_surnames
        self.oe = OrdinalEncoder()
        
        
    def fit(self, X, y=None):
        
        self.oe.fit(self.list_surnames)
        
        return self
    
    def transform(self, X, y=None):
        Xt = self.oe.transform(X)
        return Xt
    
    
def report_cv_scores(cv_scores):
    
    if isinstance(cv_scores[0], tuple):
        
        for model_name, cv_scores_ in cv_scores:
            
            mean_cv_scores = np.mean(cv_scores_)
            std_cv_scores = np.std(cv_scores_)
            
            print(f"Model used {model_name}")
            print(f"Mean scores: {mean_cv_scores}")
            print(f"Std scores: {std_cv_scores}")
            print(f"Scores: {cv_scores_}")
            print("-"*50)
        
    else:
        mean_cv_scores = np.mean(cv_scores)
        std_cv_scores = np.std(cv_scores)

        print(f"Mean scores: {mean_cv_scores}")
        print(f"Std scores: {std_cv_scores}")
        print(f"Scores: {cv_scores}")
        
class MyCustomVotingClassifier():
    
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
        
    def fit(self, X, y):
        
        for _, estimator in self.estimators:
            estimator.fit(X, y)
            
        return self
    
    
    def _predict(self, X):
        
        estimators_name = []
        estimators_predict = []
        
        for estimator_name, estimator in self.estimators:
            
            if self.voting == "hard":
                y_pred_ = estimator.predict(X) * 1
                
            else:
                y_pred_ = estimator.predict_proba(X)[:, 1]
            
            estimators_name.append(estimator_name)
            estimators_predict.append(y_pred_)
            
        return estimators_name, estimators_predict
    
    
    def _names_predictions_to_df(self, X, estimators_name, estimators_predict):
        
        df_scores = pd.DataFrame(
            data = estimators_predict
        ).T
        
        df_scores.columns = estimators_name
        df_scores.index = X.index
        
        return df_scores
        
    
    def predict(self, X):
        
        estimators_name, estimators_predict = self._predict(X=X)
            
        df_scores = self._names_predictions_to_df(
            X=X,
            estimators_name=estimators_name,
            estimators_predict=estimators_predict
        )
            
        return df_scores