import os
import numpy as np
import pandas as pd

data = pd.read_csv('student-mat.csv',sep=';')
data = data[["sex","age","guardian","traveltime","studytime","failures",
             "internet","health","absences","G1","G2","G3"]]
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#sex
label_sex = LabelEncoder()
X[:,0] = label_sex.fit_transform(X[:,0])
#guardian
label_guardian = LabelEncoder()
X[:,2] = label_guardian.fit_transform(X[:,2])
#internet
label_internet = LabelEncoder()
X[:,6] = label_internet.fit_transform(X[:,6])

#One Hot Encoder
oneHot_sex = OneHotEncoder(categorical_features=[0,2,6])
X = oneHot_sex.fit_transform(X).toarray()

#Test Train Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
#ANN
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#model
def create_model():
    model = Sequential()
    model.add(Dense(units=512,activation='relu',input_dim=15))
    model.add(Dense(units=512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,activation='relu'))
    
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

    return model
#Creating model
model = create_model()
model.summary()

#Checkpoint saving
from keras.callbacks import ModelCheckpoint

checkpoint_path = './cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,period=10)
#Fitting the training data set
model.fit(X_train,y_train,batch_size=10,epochs=100,callbacks=[cp_callback],verbose=0)

#model saving
saved_path = './my_model'
model.save(saved_path)

# Predicting a new result
y_pred = model.predict(X_test)

#Loading model only weigts
model_new = create_model()
model_new.summary()

model_new.load_weights(checkpoint_path)

#Loading saved model
model_new = create_model()
model_new.summary()

model_new.load_model(saved_path)