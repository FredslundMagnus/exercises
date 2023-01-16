import pandas as pd
from sklearn import datasets
reference_data: pd.DataFrame = datasets.load_iris(as_frame='auto').frame
current_data: pd.DataFrame = pd.read_csv('prediction_database.csv')

reference = reference_data.rename(columns=
{
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width',
}, inplace=False)
current = current_data.drop(columns=['time'], inplace=False).rename(columns={'prediction': 'target'}, inplace=False)

print(reference.head())
print(current.head())

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html('report.html') 

from evidently.test_suite import TestSuite
from evidently.tests import *
data_test = TestSuite(tests=[
    TestNumberOfMissingValues(),
    TestShareOfDriftedColumns(),
    TestNumberOfOutRangeValues(column_name='petal_width'),
    TestNumberOfOutRangeValues(column_name='petal_length'),
    TestNumberOfOutRangeValues(column_name='sepal_width'),
    TestNumberOfOutRangeValues(column_name='sepal_length'),
])
data_test.run(reference_data=reference, current_data=current)
print()
for test in data_test.as_dict()["tests"]:
    print(test["status"])