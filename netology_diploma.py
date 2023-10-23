#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


# In[3]:


hr_data = pd.read_csv('/Users/mac/Downloads/HR (1).csv')
hr_data.head()


# # Задание 2
# 
# Рассчитайте основные статистики для переменных
# (среднее,медиана,мода,мин/макс,сред.отклонение).
# 

# In[4]:


#общая статистика, переделала для удобного формата. Здесь указаны пустые значения, уникальные и общее количество.
data = pd.DataFrame()
data['uniq'] = hr_data.nunique()
data['with null'] = hr_data.isna().sum()
data['count'] = hr_data.count() - data['with null']
dtypes = hr_data.dtypes
data['type'] = dtypes
data


# In[5]:


#решила еще посмотреть количество дублей
duplicate_rows = hr_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")


# Пустых значений нет. Так как satisfaction_level, last_evaluation, average_montly_hours много уникальных значений решаю проверять являются ли они неприрывными величинами.
# 
# Также из описания к датасету и датасета непостредственно мы можем понять, что к категориальным величинам у нас относятся Work_accident,	left,	promotion_last_5years,	department,	salary
# Остальные количественные

# In[6]:


for col in hr_data.columns.values:
    if (len(hr_data[col].value_counts())> 5) and (hr_data[col].isnull().sum() > 0):
      print(col)

#нет, если бы являлись, то в результате работы кода их имена были бы в списке


# Тогда я подумала, что такое количество дублей это ошибка датафрейма. Так как каждая строка это уникальный сотрудник, то не может быть такое повторение внутри датасета

# In[7]:


hr_data = hr_data.drop_duplicates().reset_index(drop=True)
hr_data.head(5)


# In[8]:


hr_data[hr_data.duplicated() == True]


# Посмотрим статистики для количественных и категориальных величин

# In[9]:


# анализ количественных величин
hr_data.iloc[:,:5].describe().round(2).loc[['50%', 'mean', 'min', 'max', 'std']]


# In[10]:


#анализ категориальных величин (очень удобно получилось)
hr_data.iloc[:,5:10].mode()


# # Задание 3 
# 
# Рассчитайте и визуализировать корреляционную матрицу для
# количественных переменных.
# Определите две самые скоррелированные и две наименее
# скоррелированные переменные.

# In[11]:


sns.heatmap(hr_data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']].corr(), annot = True)


# **Результат:**
# 
# *   Наиболее скореллированные переменные: average_montly_hours и number_project: **0.33**
# *   ННаименее скоррелированные переменные average_montly_hours и satisfaction_level: **-0.0063**

# # Задание 4 
# 
# Рассчитайте сколько сотрудников работает в каждом
# департаменте.

# In[12]:


hr_data.groupby('department').size()


# # Задание 5 
# Показать распределение сотрудников по зарплатам. 

# In[13]:


hr_data.salary.value_counts()


# In[14]:


plt.figure(figsize = (7, 6))
hr_data['salary'].value_counts().plot(kind = 'bar')
plt.xlabel('Зарплата')
plt.ylabel('Количество сотрудников')
plt.title('Распределение сотрудников по зарплате')
plt.show()


# # Задание 6
# Показать распределение сотрудников по зарплатам в каждом
# департаменте по отдельности

# In[15]:


hr_data.pivot_table(index='department', columns='salary', aggfunc='size')


# In[16]:


plt.figure(figsize=(15, 10))
hr_data.groupby(['department', 'salary']).size().unstack().plot(kind='bar', stacked=True)
plt.xlabel('Департамент')
plt.ylabel('Количество сотрудников')
plt.title('Распределение между департаментом и зарплатой')
plt.legend(title='Зарплата')
plt.show()


# # Задание 7
# 
# Проверить гипотезу, что сотрудники с высоким окладом
# проводят на работе больше времени, чем сотрудники с низким
# окладом

# In[27]:


pivot_table = hr_data.pivot_table(index='salary', values='average_montly_hours', aggfunc='mean')
print(pivot_table)

pivot_table.plot(kind='bar', legend=False)
plt.xlabel('Уровень зарплаты')
plt.ylabel('Среднее количество часов на рабочем месте в месяц')
plt.title('Сравнение среднего количества часов на рабочем месте в месяц по уровню зарплаты')
plt.show()


# Визуально разницы в рабочих часах между сотрудниками с высоким и низким окладом нет, однако проверим это с помощью проверки гипотезы Предположим, что Н_0 = в среднем проведенное на работе время одинаково для всех работников

# In[28]:


salary_low = hr_data[hr_data['salary'] == 'low']['average_montly_hours']
salary_high = hr_data[hr_data['salary'] == 'high']['average_montly_hours']

result = stats.ttest_ind(salary_low, salary_high, equal_var=False)
a = 0.05
if (result.pvalue < a):
    print('Отвергаем нулевую гипотезу')
else:
    print('Нулевая гипотеза не отвергается')


# Можно с уверенностью сказать, что разницы в работе между сотрудниками с маленькой и большой зарплатой нет

# # Задание 8
# Рассчитать следующие показатели среди уволившихся и не
# уволившихся сотрудников (по отдельности):
# 
# 
# *   Доля сотрудников с повышением за последние 5 лет
# 
# *   Средняя степень удовлетворенности
# *   Среднее количество проектов
# 

# In[32]:


promoted_left = hr_data[hr_data['left'] == 1]['promotion_last_5years'].mean()
promoted_not_left = hr_data[hr_data['left'] == 0]['promotion_last_5years'].mean()
satisfaction_left = hr_data[hr_data['left'] == 1]['satisfaction_level'].mean()
satisfaction_not_left = hr_data[hr_data['left'] == 0]['satisfaction_level'].mean()
projects_left = hr_data[hr_data['left'] == 1]['number_project'].mean()
projects_not_left = hr_data[hr_data['left'] == 0]['number_project'].mean()


# In[40]:


results = pd.DataFrame(columns=['Повышение за последние 5 лет', 'Уровень удовлетворенности', 'Количество проектов'])
results.loc['Ушли'] = [promoted_left, satisfaction_left, projects_left]
results.loc['Не ушли'] = [promoted_not_left, satisfaction_not_left, projects_not_left]
results


# # Задание 9
# 
# Разделить данные на тестовую и обучающую выборки
# Построить модель LDA, предсказывающую уволился ли
# сотрудник на основе имеющихся факторов (кроме department и
# salary)
# Оценить качество модели на тестовой выборки

# In[41]:


# Разделение данных на признаки (X) и целевую переменную (y)
X = hr_data.drop(['left', 'department', 'salary'], axis=1)
y = hr_data['left']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[43]:


# Построение модели LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Оценка качества модели на тестовой выборке
accuracy = lda.score(X_test, y_test)
print("Accuracy:", accuracy)


# 0.83 - это высокий показатель. Значит, что модель репрезентативная

# # Задание 10
# 
# https://github.com/Natadelaer/netology_diploma
