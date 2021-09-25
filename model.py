import pandas as pd
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
import pandas as pd
import keras
''' Loading data '''
df = pd.read_csv(r"C:\Users\PRAMILA\Downloads\Bangaluru_Wind Generation.csv")
#print(df)
df=df.drop(columns=['Timeseries'])
''' Cleaning Data '''
#dataframe.drop['Date'].values
df['Power Generated\n(kw)'].replace(0, np.nan, inplace=True)
df['Power Generated\n(kw)'].fillna(method='ffill', inplace=True)


X = df.drop(columns=['Power Generated\n(kw)'])
Y = df[['Power Generated\n(kw)']]
X=np.array(X).reshape(-1,1,4)
Y=np.array(Y).reshape(-1,1,1)



model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu',input_shape=(-1,1,4))))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
model.fit(X, Y,epochs=1,callbacks=[keras.callbacks.EarlyStopping(patience=3)])
test_data = np.array([[27.99,1066,2.57,260]])
o=model.predict(test_data.reshape(-1,1,4), batch_size=1)
print(o)

# Saving model to disk
models=model.save('model.h5')



