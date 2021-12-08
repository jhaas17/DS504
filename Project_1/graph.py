import pandas as pd
import random
import matplotlib.pyplot as plt

# df = pd.read_csv('100samp')
#
# df2 = pd.read_csv('100samp2')
#
# real = df.append(df2, ignore_index=True)
# real.drop(1)
# num_samples = [10, 20, 30, 40, 50, 75, 100, 200]
# print(real.sample(n=10)['100'])
# for num in num_samples:
#     df1 = real.sample(n=num)
#     df = pd.concat([df, df1['100']], axis=1)
#     df.to_csv(('df' + str(num)))
#     print("Finished " + str(num) + " samples! ")
df200 = pd.read_csv('dfFULL11200')
df100 = pd.read_csv('dfFULL100')
df = pd.read_csv('dfFULL200')
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(df)
print(df200['200'][:5])
print(df['200'][5:])
df['200'][5:] = df200['200'][:5]
df100['200'] = df['200']
df=df100
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
# print(df)
df.to_csv('FULLFinal')
# print(df)
bplot = df.boxplot()
bplot.set_ylabel('Active Github User Estimation')
bplot.set_xlabel('Number of Samples')

plt.savefig('Full200')
plt.show()
print(df['200'].mean())