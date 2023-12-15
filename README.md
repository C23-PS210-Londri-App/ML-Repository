# ML-Repository
Repository for ML-Learning Path

## How To Use
### First Load the Model and Scaler(Pickle)
```python
import tensorflow as tf
from keras import load_model

model_path = "your_model_path"
model = load_model(model_path)

with open('name_of_the_scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
```



