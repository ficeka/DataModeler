"""
Quick assignment demonstrating use of pandas and tensorflow to predict a binary category using a very small neural net.

"""


from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):
        '''
        Initialize the DataModeler as necessary.
        '''
        self.train_df = sample_df
        self.model = None

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Prepare a dataframe so it contains only the columns to model and having suitable types.
        If the argument is None, work on the training data passed in the constructor.
        '''
        if oos_df is None: #If the argument is None, work on the training data passed in the constructor.
            oos_df = self.train_df
        if 'customer_id' in oos_df.columns: # Drop customer id column in-place if it exists
            oos_df.drop(labels=['customer_id'], axis =1, inplace=True)
        oos_df = oos_df.astype({'amount': 'float64'}, copy=False)
        oos_df['transaction_date'] = oos_df['transaction_date'].to_numpy(dtype='datetime64[ns]', na_value=None).astype('float64', copy=False)
        oos_df['transaction_date'] = oos_df['transaction_date'].where(oos_df['transaction_date'] > 0, np.nan)

        self.train_df = oos_df
        return oos_df

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Fill any missing values with the appropriate mean (average) value.
        If the argument is None, work on the training data passed in the constructor.
        '''
        if oos_df is None:
            oos_df = self.train_df
        columns_to_use = ['amount', 'transaction_date']
        oos_df[columns_to_use] = oos_df[columns_to_use].fillna(oos_df[columns_to_use].mean())

    def fit(self) -> None:
        '''
        Fit the model of your choice on the training data passed in the constructor, assuming it has
        been prepared by the functions prepare_data and impute_missing
        '''
        target = self.train_df['outcome']
        features = self.train_df[['amount', 'transaction_date']]
        tf.convert_to_tensor(features)
        normalizer = tf.keras.layers.Normalization(axis=-1, input_shape=(None,2))
        normalizer.adapt(features)

        def get_basic_model():
            model = tf.keras.Sequential([
                normalizer,

                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(1),
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            return model
        self.model = get_basic_model()
        self.model.fit(features, target, epochs=1000)



    def model_summary(self) -> str:
        '''
        Create a short summary of the model you have fit.
        '''
        return "This is a very basic model. It uses a normalizer layer followed by two 4-neuron Dense layers"

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series[bool]:
        '''
        Make a set of predictions with your model. Assume the data has been prepared by the
        functions prepare_data and impute_missing.
        If the argument is None, work on the training data passed in the constructor.
        '''
        if not oos_df:
            oos_df = self.train_df
        features = oos_df[['amount', 'transaction_date']]
        tf.convert_to_tensor(features)
        return pd.Series([x[0]>0 for x in self.model.predict(features)])

    def save(self, path: str) -> None:
        '''
        Save the DataModeler so it can be re-used.
        '''
        file_name = path + '.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            print(f'Object successfully saved to "{file_name}"')

    @staticmethod
    def load(path: str) -> DataModeler:
        '''
        Reload the DataModeler from the saved state so it can be re-used.
        '''
        with open(path +'.pkl', 'rb') as picklefile:
            DataModeler = pickle.load(picklefile)
        return DataModeler


#################################################################################


transact_train_sample = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            '2022-01-01',
            '2022-08-01',
            None,
            '2022-12-01',
            '2022-02-01',
            None,
            '2022-02-01',
            '2022-01-01',
            '2022-11-01',
            '2022-01-01'
        ],
        "outcome" : [False, True, True, True, False, False, True, True, True, False]
    }
)


print(f"Training sample:\n{transact_train_sample}\n")

# <Expected Output>
# Training sample:
#    customer_id  amount transaction_date  outcome
# 0           11     1.0       2022-01-01    False
# 1           12     3.0       2022-08-01     True
# 2           13    12.0             None     True
# 3           14     6.0       2022-12-01     True
# 4           15     0.5       2022-02-01    False
# 5           16     0.2             None    False
# 6           17     NaN       2022-02-01     True
# 7           18     5.0       2022-01-01     True
# 8           19     NaN       2022-11-01     True
# 9           20     3.0       2022-01-01    False


print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

# <Expected Output>
# Current dtypes:
# customer_id           int64
# amount              float64
# transaction_date     object
# outcome                bool
# dtype: object

transactions_modeler = DataModeler(transact_train_sample) #1. Construct DataModeler

transactions_modeler.prepare_data() # 2. call method prepare_data to change column type of amount and drop cust_id

print(f"Changed columns to:\n{transactions_modeler.train_df.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

transactions_modeler.impute_missing() #3. impute missing values for all columns

print(f"Imputed missing as mean:\n{transactions_modeler.train_df}\n")

# <Expected Output>
# Imputed missing as mean:
#               amount  transaction_date
# customer_id
# 11            1.0000      1.640995e+18
# 12            3.0000      1.659312e+18
# 13           12.0000      1.650845e+18
# 14            6.0000      1.669853e+18
# 15            0.5000      1.643674e+18
# 16            0.2000      1.650845e+18
# 17            3.8375      1.643674e+18
# 18            5.0000      1.640995e+18
# 19            3.8375      1.667261e+18
# 20            3.0000      1.640995e+18


print("Fitting  model")
transactions_modeler.fit()

print(f"Fit model:\n{transactions_modeler.model_summary()}\n")

# <Expected Output>
# Fitting  model
# Fit model:


in_sample_predictions = transactions_modeler.predict()
print(f"Predicted on training sample: {in_sample_predictions}\n")
print(f'Accuracy = {sum(in_sample_predictions ==  [False, True, True, True, False, False, True, True, True, False])/.1}%')

# <Expected Output>
# Predicting on training sample [False  True  True  True False False True  True  True False]
# Accuracy = 100.0%

transactions_modeler.save("transact_modeler")
loaded_modeler = DataModeler.load("transact_modeler")

print(f"Loaded DataModeler sample df:\n{loaded_modeler.model_summary()}\n")

# <Expected Output>
# Loaded DataModeler sample df:

transact_test_sample = pd.DataFrame(
    {
        "customer_id": [21, 22, 23, 24, 25],
        "amount": [0.5, np.nan, 8, 3, 2],
        "transaction_date": [
            '2022-02-01',
            '2022-11-01',
            '2022-06-01',
            None,
            '2022-02-01'
        ]
    }
)

adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)

print(f"Changed columns to:\n{adjusted_test_sample.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)

print(f"Imputed missing as mean:\n{filled_test_sample}\n")

# <Expected Output>
# Imputed missing as mean:
#              amount  transaction_date
# customer_id
# 21           0.5000      1.643674e+18
# 22           3.8375      1.667261e+18
# 23           8.0000      1.654042e+18
# 24           3.0000      1.650845e+18
# 25           2.0000      1.643674e+18

oos_predictions = transactions_modeler.predict(filled_test_sample)
print(f"Predicted on out of sample data: {oos_predictions}\n")
print(f'Accuracy = {sum(oos_predictions == [False, True, True, False, False])/.05}%')

# <Expected Output>
# Predicted on out of sample data: [False True True False False] ([0 1 1 0 0])
# Accuracy = 100.0%

