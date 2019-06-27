import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import numpy as np
 
url = './railwaytraffic.csv'
ass_data = requests.get(url).content
 
df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  # python2使用StringIO.StringIO
 
data = np.array(df['railwaytraffic_current value(ten thousand)'])
# normalize
normalized_data = (data - np.mean(data)) / np.std(data)
 
plt.figure()
plt.plot(data)
plt.show()
