# This simple Auto Machine learning BlackBox
---
1. automate normalize data
2. automate dimension reduction
3. automate model create 
4. automate find best paramters for your data
## How to use it?

```python
import model
model.find(X,y)
```
### what is parameter?
```python
# all parametrs
model.find(X, Y, test_size= .2, preprocessing_type= 0, dimension_reduction_type= 0, models_type= 0, verbose= 1, normalize_validation_data= False,  random_state= 42)
```
- X -- input features
- Y -- output lables
- split_type  -- how data splite between train and test
- preprocessing_type  -- kind of nomalizer, 0  = without any processing, 1  = full
- dimension_reduction_type  -- kind of dimension reduction, 0  = without any dimension reduction, 1  = full dimension reduction
- models_type  -- model types, 0  = simple models , 1  = full models
- verbose  --to print process , verbose = 1 print per model else not print result per model
