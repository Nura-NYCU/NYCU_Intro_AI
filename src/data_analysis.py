import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/covid_fake_news.csv')

print(df.head(), end = '\n\n')

outcomes = df['outcome']

counts = outcomes.value_counts()

plt.figure()
plt.title('Data Count')
ax = counts.plot(kind='bar')
ax.patches[1].set_facecolor('red')
plt.ylabel('Counts')
plt.xticks(ticks = [0,1], labels=['Zero', 'One'], rotation = 0)
plt.show()
