import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('animals_train.csv')
class_mapping = pd.read_csv('animal_classes.csv')

class_map = dict(zip(class_mapping['Class_Number'], class_mapping['Class_Type']))
#train_data['class_type'] = train_data['class_number'].map(class_map)

X_train = train_data.drop(columns=['class_number'])
y_train = train_data['class_number']

model = RandomForestClassifier()
model.fit(X_train, y_train)

test_data = pd.read_csv('animals_test.csv')

X_test = test_data.drop(columns=['animal_name'])

predictions = model.predict(X_test)

predicted_class_types = [class_map.get(pred, 'Unknown') for pred in predictions]

output_df = pd.DataFrame({'animal_name': test_data['animal_name'], 'prediction': predicted_class_types})
output_df.to_csv('predictions.csv', index=False)