#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


url = 'https://raw.githubusercontent.com/Sketchjar/MachineLearningHD/main/master.csv'

originaldata = pd.read_csv(url)
originaldata.head()


# In[48]:


url1 = 'https://raw.githubusercontent.com/Sketchjar/MachineLearningHD/main/suicidedataextrafestures.csv'
extra_data = pd.read_csv(url1)

extra_data.head().T


# In[49]:


extra_data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population','suicidesper100k', 'country-year', 'yearlyHDI',
    'GDPpyear', 'GDPpcapita', 'generation', 'suicide%', 'Internetusers', 'Expenses', 'employeecompensation','Unemployment', 'Physiciansp1000', 'Legalrights', 'Laborforcetotal','Lifeexpectancy', 'Mobilesubscriptionsp100','Refugees', 'Selfemployed', 'electricityacess', 'secondarycompletion']


# In[50]:


extra_data.shape


# In[51]:


extra_data.head()


# In[52]:


originaldata.head()


# In[53]:


originaldata.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population','suicidesper100k','country-year', 'yearlyHDI', 'GDPpyear', 'GDPpcapita', 'generation']


# In[54]:


extra_data.head()


# In[55]:


#fixing and cleaning the original data  
originaldata['GDPpyear'] = originaldata.apply(lambda x: float(x['GDPpyear'].replace(',', '')), axis=1)
originaldata.sex.astype('category')


# In[56]:


countrynames = ['Argentina','Armenia','Australia',    'Austria',
    'Belgium',    'Brazil',    'Bulgaria',    'Canada',    'Chile',    'Colombia',    'Croatia',    'Cuba',    'Czech Republic',    'Denmark',
    'Finland',    'France',    'Germany',    'Greece',    'Hungary',    'Iceland',   'Ireland', 'Israel','Italy','Japan','Mexico', 'Netherlands','New Zealand','Norway','Poland', 'Portugal','Romania','Russian Federation','South Africa', 'Spain','Sweden', 'Switzerland','Thailand', 'Turkmenistan','Ukraine','United Kingdom', 'United States']


# In[57]:


df1 = extra_data.copy()
df = df1.iloc[np.where(df1.country == countrynames[0])]
for i, x in enumerate(countrynames[1:]):
    df = df.append(df1.iloc[np.where(df1.country == x)])
    
print(df)


# In[58]:


col = plt.cm.Spectral(np.linspace(0, 1, 20))

plt.figure(figsize=(8, 6))

agedistf = pd.DataFrame(df.groupby('sex').get_group('female').groupby('age').suicides_no.sum())
print(agedistf)

agedistm = pd.DataFrame(df.groupby('sex').get_group('male').groupby('age').suicides_no.sum())
print(agedistm)
#-----------------------------------------------------
print(agedistm.suicides_no)

plt.bar(agedistm.index, agedistm.suicides_no, color=col[18])
plt.bar(agedistf.index, agedistf.suicides_no, color=col[7])
plt.legend(['male', 'female'], fontsize=16)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Suicides per 100K', fontsize=14)


# In[59]:


plt.figure(figsize=(12, 15))


plt.subplot(211)
df.groupby(['country']).suicidesper100k.mean().nlargest(10).plot(kind='barh', color=col)
plt.xlabel('Average Suicides/100k', size=20)
plt.ylabel('Country', fontsize=20)
plt.title('Top 10 countries', fontsize=30)

plt.subplot(212)
df.groupby(['country']).suicides_no.mean().nlargest(10).plot(kind='barh', color=col)
plt.xlabel('Average Suicides_no', size=20)
plt.ylabel('Country', fontsize=20);


# In[60]:


import seaborn as sns
plt.figure(figsize=(10, 16))

plt.subplot(311)

sns.barplot(x='sex', y='population', hue='age', data=df, palette="Blues")
plt.xticks(ha='right', fontsize=20)
plt.ylabel('Population', fontsize=20)
plt.xlabel('Sex', fontsize=20)
plt.legend(fontsize=14, loc='best')

plt.subplot(313)

sns.barplot(x='sex', y='suicidesper100k', hue='age', data=df,palette="Blues")
plt.xticks(ha='right', fontsize=20);
plt.ylabel('suicidesper100k',fontsize=20);
plt.xlabel('Sex',fontsize=20);
plt.legend(fontsize=14);

plt.subplot(312)
sns.barplot(x='sex', y='suicides_no', hue='age', data=df, palette="Blues")
plt.xticks(ha='right', fontsize=20)
plt.ylabel('suicides incidences', fontsize=20)
plt.xlabel('Sex', fontsize=20)
plt.legend(fontsize=14)


# In[61]:


year = originaldata.groupby('year').year.unique()

plt.figure(figsize=(6, 5))

totalpyear = pd.DataFrame(originaldata.groupby('year').suicides_no.sum())
print(totalpyear)
plt.plot(year.index[0:31], totalpyear[0:31], color=col[18])
plt.xlabel('year', fontsize=14)
plt.ylabel('Total number of suicides in the world', fontsize=14)


# In[62]:


plt.figure(figsize=(20, 8))
plt.subplot(121)
plt.hist(df.suicidesper100k, bins=30, color=col[18])
plt.xlabel('Suicides per 100K of population', fontsize=14)
plt.ylabel('count', fontsize=14)

plt.subplot(122)
plt.hist(df.GDPpcapita, bins=30, color=col[7])
plt.xlabel('GDP', fontsize=14)
plt.ylabel('count', fontsize=14)


# In[63]:


features = ['country', 'year', 'GDPpyear', 'GDPpcapita', 'employeecompensation', 'Unemployment',
            'Lifeexpectancy', 'Refugees', 'Selfemployed', 'Internetusers']

total = df[features].groupby('country').get_group(countrynames[0]).groupby('year').mean()
total['Suicides'] = df[['country', 'year', 'suicidesper100k']].groupby('country').get_group(countrynames[0]).groupby('year').sum()
print(total)
total['population'] = df[['country', 'year', 'population']].groupby('country').get_group(countrynames[0]).groupby('year').sum()

total['country'] = countrynames[0]

for i, x in enumerate(countrynames[1:]):
    suicides = df[features].groupby('country').get_group(x).groupby('year').mean()
    suicides['Suicides'] = df[['country', 'year', 'suicidesper100k']].groupby('country').get_group(x).groupby('year').sum()
    total['population'] = df[['country', 'year', 'population']].groupby('country').get_group(x).groupby('year').sum()
  
    suicides['country'] = x
    total = total.append(suicides)

total.reset_index(inplace=True)
sort = True


# In[64]:


totalfeatures = ['country', 'year', 'GDPpyear', 'GDPpcapita', 'employeecompensation', 'Unemployment','Lifeexpectancy', 'Refugees', 'Selfemployed', 'Internetusers', 'population']


# In[65]:


totalfeatures = ['country', 'year', 'GDPpyear', 'GDPpcapita', 'employeecompensation', 'Unemployment','Lifeexpectancy', 'Refugees', 'Selfemployed', 'Internetusers', 'population']


# In[66]:


plt.figure(figsize=(20, 8))
plt.subplot(121)
sns.distplot(total.Suicides, bins=15)
plt.xlabel('total Suicides (summed over sex and age group) per 100K of population', fontsize=14)

plt.subplot(122)
plt.hist(total.GDPpcapita, bins=30, color=col[7])
plt.xlabel('GDP', fontsize=14)


# In[67]:


plt.figure(figsize=(8, 5))

suicides = df[['year', 'GDPpyear', 'Selfemployed', 'Unemployment', 'Lifeexpectancy']].groupby('year').mean()
suicides['Suicides'] = df[['country', 'year', 'suicidesper100k']].groupby('year').sum()

plt.plot(suicides.index, suicides.GDPpyear/suicides.GDPpyear.max(), color=col[1])
plt.plot(suicides.index, suicides.Unemployment/suicides.Unemployment.max(), color=col[7])
plt.plot(suicides.index, suicides.Lifeexpectancy/suicides.Lifeexpectancy.max(), color=col[15])
plt.plot(suicides.index, suicides.Suicides/suicides.Suicides.max(), color=col[17])
plt.legend(['global average GDPpyear', 'global average Unemployment', 'global average Life expectancy', 'Total suicides per 100k'], fontsize=14, loc='best')
plt.ylabel('Normalized', fontsize=14)
plt.xlabel('year', fontsize=14)


# In[68]:


corr = total.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=0.2, cbar_kws={"shrink": 0.8});


# In[69]:


total.Internetusers.fillna(total.Internetusers. min(), inplace=True)
total.Refugees.fillna(8, inplace=True)
total.employeecompensation.fillna(total.employeecompensation.mean(), inplace=True)
total.population.fillna(total.population.mean(), inplace=True)


# In[70]:


total['risk'] = total.Suicides.copy()

total['risk'] = np.where(total.risk < total.Suicides.mean(), 0, 1)


# In[71]:


plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.hist(total.risk, color=col[6])
plt.ylabel('counts', fontsize=14)
plt.xlabel('Suicide risk', fontsize=14)

plt.subplot(122)
sns.distplot(total.Suicides[total.risk == 0], bins=10)
sns.distplot(total.Suicides[total.risk == 1], bins=20)  
plt.xlabel('Suicides', fontsize=14)


# In[72]:


d=total[totalfeatures]
print(total[totalfeatures])
x=total[totalfeatures]
p=x.drop('country',axis=1)
print(p.columns)
print(p)


# In[73]:


from sklearn.preprocessing import StandardScaler
# Preparing data for modeling

X = np.asarray(p)
y = np.asarray(total['risk'])


# Applying standard scaler on data, since ML algorithms work with the assumption that the data is normally distributed

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[74]:


from sklearn.model_selection import train_test_split
# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# In[75]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


DT = DecisionTreeClassifier(criterion="entropy", max_depth=7, max_leaf_nodes=30)
DT = DT.fit(X_train, y_train)
ydthat = DT.predict(X_test)

print('******************Decision Tree classifier**************')

print('Accuracy =', DT.score(X_test, y_test))
print('Train Accuracy=', DT.score(X_train, y_train))
print('CM\n', confusion_matrix(y_test, ydthat))
print('classification report for decision tree\n', classification_report(y_test, ydthat))
print('# of leaves', DT.get_n_leaves(), '\n Depth', DT.get_depth())


DTfeat_importance = DT.feature_importances_
DTfeat_importance = pd.DataFrame([totalfeatures, DT.feature_importances_]).T


print(DTfeat_importance.sort_values(by=1, ascending=False))
print('\n# of features= ', DT.n_features_)


# In[76]:


import seaborn as sn
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,ydthat)
sn.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Positive", "Negative"] , yticklabels = ["True", "False"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[77]:


from sklearn.tree import plot_tree, export_text
plt.figure(figsize=(80,20))
plot_tree(df, feature_names=X_train.columns, max_depth=2, filled=True);


# In[78]:


X_train


# In[79]:


originaldata.columns


# In[80]:


plot_tree( df,
    max_depth=2,
    feature_names=None,
    class_names=None)


# In[ ]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree


# In[ ]:


print(X_scaled)


# In[34]:


plt.figure(figsize=(80,20))
plot_tree(total, feature_names=X_scaled.columns, max_depth=2, filled=True);


# In[35]:


X_train.shape


# In[36]:


x.shape


# In[37]:


print(export_text(DT, feature_names=list(x.columns)))


# In[38]:


total.feature_names


# In[39]:


X_scaled


# In[40]:


s=total['risk']


# In[41]:


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DT, 
                   feature_names=p.columns,  
                   class_names=s,
                   filled=True)


# In[42]:


total.risk


# In[43]:


model = DecisionTreeClassifier(random_state=42)
model.fit


# In[44]:



plt.figure(figsize=(20,10))
plot_tree(DT, feature_names=p.columns, max_depth=2, filled=True);


# In[45]:


tree_text = export_text(DT, max_depth=4, feature_names=list(p.columns))
print(tree_text[:5000])


# In[ ]:


p.columns


# In[ ]:


total.shape


# In[ ]:


total.head()


# In[ ]:




