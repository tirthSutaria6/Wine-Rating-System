import pandas
import numpy as np
from statsmodels import robust

url = "winequality-red.csv"
names = ['Fixed Acidity','Volatile Acidity','Citric Acid','Residual Sugar','Chlorides','Free SO2','Total SO2','Density','pH','Sulphates','ALC by Vol','Quality']
data = pandas.read_csv(url, names=names)
print np.mean(data)
print "\n"
print "Median of all Attributes:"
print (np.median(data, axis=0))
print "\n"
print "Standard Deviation of all Attributes:"
print np.std(data,axis=0)

print "\n"
mad = robust.mad(data, axis=0)
print "MAD of the attributes given is: "
print mad
print "\n"
max_data = np.max(data,axis=0)
min_data = np.min(data,axis=0)

print "Maximum and minimum data points are given below:"
print max_data
print "\n"
print min_data

print data['Quality']