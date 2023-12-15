# ML-Repository
Repository for ML-Learning Path

## How To Use The Model
### First Load the Model and Scaler(Pickle)
```python
import tensorflow as tf
from keras import load_model

model_path = "your_model_path/model_name.h5"
model = load_model(model_path)

with open('scaler_path/name_of_the_scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
```

### Prepare the New Data Point
```python
new_data_point = np.array([-7.87639, 110.35889, 1, 0, 1, 1, 1])
#This is example of the data point, consisting of [latitude,longitude,kategori1,kategori2,kategori3,kategori4,kategori5]

#We transform the longitude and latitude with Scaler
new_data_point[:2] = loaded_scaler.transform(new_data_point[:2].reshape(1,-1))

#Reshape it before predicting
new_data_point = new_data_point.reshape(1, -1)
```
### Making Prediction
```python
predictions = loaded_model.predict(new_data_point)

#Get the probabilities of class predictions
class_probabilities = predictions[0]

#Sorting it
sorted_indices = np.argsort(class_probabilities1)[::-1]
next_most_likely_indices = sorted_indices1[1:16] #Here it return 15 places id, you can change it accordingly

#Inside the next_most_likely_indices
[607 608 644 623 645 613 664 636 621 651 649 618 652 655 640] #You can search this id in the database to get data of laundry places
```
