# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 07:38:29 2020

@author: Jovana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sb
from scipy.stats import norm
import seaborn as sb
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
from sklearn.linear_model import RidgeCV

pd.set_option('display.float_format', lambda x: '%.2f' % x)

data_set = pd.read_csv("PRSA_Data_Tiantan_20130301-20170228.csv")

prvih_pet = data_set.head()

#iz baze izbacujemo kolone koje se odnose na redni broj merenja i naziv stanice (redni broj merenja je jedinstven za svako merenje, a sva merenja su vrsena na jednoj stanici)
data_set.drop(['No', 'station'], axis = 1, inplace = True)

print("Broj uzoraka i obelezja je: ", data_set.shape)

#provera da li je obelezje kategoricko ili numericko
print(data_set['year'].unique())
print(data_set['month'].unique())
print(data_set['day'].unique())
print(data_set['hour'].unique())
print(data_set['wd'].unique())
print(data_set['CO'].unique())
#print(data_set['station'].unique())

#dimenzije baze
print(data_set.shape)

#tipovi podataka u bazi
print(data_set.dtypes)

print(round(data_set.isnull().sum()/len(data_set.index), 2)*100)

data_set.isnull().sum()

print("Atribut PM2.5 ima: ", data_set['PM2.5'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['PM2.5'].isna().sum()/len(data_set)*100, '%')
print("Atribut PM10 ima: ", data_set['PM10'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['PM10'].isna().sum()/len(data_set)*100, '%')
print("Atribut SO2 ima: ", data_set['SO2'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['SO2'].isna().sum()/len(data_set)*100, '%')
print("Atribut NO2 ima: ", data_set['NO2'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['NO2'].isna().sum()/len(data_set)*100, '%')
print("Atribut C0 ima: ", data_set['CO'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['CO'].isna().sum()/len(data_set)*100, '%')
print("Atribut O3 ima: ", data_set['O3'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['O3'].isna().sum()/len(data_set)*100, '%')
print("Atribut TEMP ima: ", data_set['TEMP'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['TEMP'].isna().sum()/len(data_set)*100, '%')
print("Atribut PRES ima: ", data_set['PRES'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['PRES'].isna().sum()/len(data_set)*100, '%')
print("Atribut DEWP ima: ", data_set['DEWP'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['DEWP'].isna().sum()/len(data_set)*100, '%')
print("Atribut wd ima: ", data_set['wd'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['wd'].isna().sum()/len(data_set)*100, '%')
print("Atribut WSPM ima: ", data_set['WSPM'].isnull().sum(), "nedostajućih vrednosti, odnosno:" , data_set['WSPM'].isna().sum()/len(data_set)*100, '%')

#pretvaranje vrednosti kategorickog obelezja u numericke vrednosti
data_set.loc[data_set['wd']=='N','wd']= 1
data_set.loc[data_set['wd']=='NNE','wd']= 2
data_set.loc[data_set['wd']=='NE','wd']= 3
data_set.loc[data_set['wd']=='ENE','wd']= 4
data_set.loc[data_set['wd']=='E','wd']= 5
data_set.loc[data_set['wd']=='ESE','wd']= 6
data_set.loc[data_set['wd']=='SE','wd']= 7
data_set.loc[data_set['wd']=='SSE','wd']= 8
data_set.loc[data_set['wd']=='S','wd']= 9
data_set.loc[data_set['wd']=='SSW','wd']= 10
data_set.loc[data_set['wd']=='SW','wd']=11
data_set.loc[data_set['wd']=='WSW','wd']= 12
data_set.loc[data_set['wd']=='W','wd']=13
data_set.loc[data_set['wd']=='WNW','wd']=14
data_set.loc[data_set['wd']=='NW','wd']=15
data_set.loc[data_set['wd']=='NNW','wd']=16


#zamena nedostajućih vrednosti metodom ffill
data_set['PM2.5'].fillna(method = 'ffill', inplace = True)
data_set['PM10'].fillna(method = 'ffill', inplace = True)
data_set['SO2'].fillna(method = 'ffill', inplace = True)
data_set['NO2'].fillna(method = 'ffill', inplace = True)
data_set['CO'].fillna(method = 'ffill', inplace = True)
data_set['O3'].fillna(method = 'ffill', inplace = True)

#izostavljanje uzoraka koja imaju obelezja ciji je broj nedostajućih vrednosti čiji je procenat manji od 1%
data_set.dropna(axis = 0, inplace = True)


#pomoćni kod za određivanje načina zamene nedostajućih vrednosti
pm2_nan = data_set[['PM2.5', 'year', 'month', 'hour', 'day']]
is_NaNPM2 = pm2_nan.isnull()
row_has_nan = is_NaNPM2.any(axis = 1)
row_with_NaN = pm2_nan[row_has_nan]
row_with_NaN = row_with_NaN.sort_values(by = (['year', 'month', 'day', 'hour']), ascending=True)

#statisticki parametri
#_______________________________________________________________________________________________________
statisticki_parametri = data_set.describe()

data_set_year = data_set.set_index('year')

plt.boxplot([data_set["PM2.5"], data_set["PM10"], data_set["NO2"], data_set["SO2"], data_set["O3"]])
plt.ylabel("Koncentracija čestica")
plt.yticks(np.arange(0, 1000, 100))
plt.xticks([1, 2, 3, 4, 5], ["PM2.5", "PM10", "NO2", "SO2", "O3"])
plt.grid()

plt.boxplot([data_set_year["CO"]])
plt.yticks(np.arange(0, 10000, 1000))
plt.xticks([1], ["CO"])
plt.grid()

data_set["CO"].describe()

O3_outliers = data_set[data_set["O3"] == 674]
CO_outliers = data_set[data_set["CO"] == 10000]

#kolicina padavina po mesecima
data_set2013p = data_set[data_set['year'] == 2013].groupby('month').mean()
plt.plot(data_set2013p['RAIN'], label = '2013')
data_set2014p = data_set[data_set['year'] == 2014].groupby('month').mean()
plt.plot(data_set2014p['RAIN'], label = '2014')
data_set2015p = data_set[data_set['year'] == 2015].groupby('month').mean()
plt.plot(data_set2015p['RAIN'], label = '2015')
data_set2016p = data_set[data_set['year'] == 2016].groupby('month').mean()
plt.plot(data_set2016p['RAIN'], label = '2016')
data_set2017p = data_set[data_set['year'] == 2017].groupby('month').mean()
plt.plot(data_set2017p['RAIN'], label = '2017')
plt.legend()
plt.xticks(np.arange(0, 13, 1))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["Jan", "Feb", "Mar", "Ap", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])


#_____________________________
#detaljna analiza obelezja NO2

data_set_month = data_set.set_index('month')

#statisticke velicine za obelezje NO2
data_NO2 = data_set["NO2"]
data_NO2.describe()

#koncentracija NO2 po mjesecima svake godine
plt.figure(figsize = (20, 10))
plt.boxplot([data_set_month.loc[1, "NO2"], data_set_month.loc[2, "NO2"], data_set_month.loc[3, "NO2"],data_set_month.loc[4, "NO2"], 
             data_set_month.loc[5, "NO2"], data_set_month.loc[6, "NO2"], data_set_month.loc[7, "NO2"], data_set_month.loc[8, "NO2"], data_set_month.loc[9, "NO2"]
             ,data_set_month.loc[10, "NO2"], data_set_month.loc[11, "NO2"], data_set_month.loc[12, "NO2"]])
plt.ylabel("Koncentracija čestica NO2 u vazduhu")
plt.yticks(np.arange(0, 300, 10))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["Jan", "Feb", "Mar", "Ap", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.grid()
#zakljucak: u januaru i decembru svake godine povecana vrednost NO2, tad su se javljale i najvece vrednosti outlijera


#koncentracija NO2 u godinima 2013, 2015 i 2017
data_set_year = data_set.set_index('year')
plt.hist(data_set_year.loc["2013", "NO2"], density = True, alpha = 0.5, label = "2013", bins = 20)
plt.hist(data_set_year.loc["2015", "NO2"], density = True, alpha = 0.5, label = "2015", bins = 20)
plt.hist(data_set_year.loc["2017", "NO2"], density = True, alpha = 0.5, label = "2017", bins = 20)
plt.ylabel("Verovatnoca")
plt.xlabel("NO2")
plt.legend()
#zakljucak: za sve navedene godine vazi da je mala verovatnoca da je koncentracija NO2 veca od 150
#ideja je da se uporede vrednosti NO2 na svake dve godine


#koncentracija NO2 po mjesecima razlicitih godina
data_set2013m = data_set[data_set['year'] == 2013].groupby('month').mean()
plt.plot(data_set2013m['NO2'], label = '2013')
data_set2014m = data_set[data_set['year'] == 2014].groupby('month').mean()
plt.plot(data_set2014m['NO2'], label = '2014')
data_set2015m = data_set[data_set['year'] == 2015].groupby('month').mean()
plt.plot(data_set2015m['NO2'], label = '2015')
data_set2016m = data_set[data_set['year'] == 2016].groupby('month').mean()
plt.plot(data_set2016m['NO2'], label = '2016')
data_set2017m = data_set[data_set['year'] == 2017].groupby('month').mean()
plt.plot(data_set2017m['NO2'], label = '2017')
plt.xlabel("Prosečna koncentracija")
plt.xticks(np.arange(0, 13, 1))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["Jan", "Feb", "Mar", "Ap", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])


df_year = data_set.set_index('year')
df_year.head()

#koncentracija NO2 po satu tokom svih godina
plt.figure(figsize = (15, 10))
no2_po_satu = df_year[['NO2', 'hour']].groupby(["hour"]).mean()
plt.plot(no2_po_satu)
plt.xticks(np.arange(0, 24, 1))
plt.ylabel(np.arange(0, 60, 5))
plt.grid()

#koeficijenti asimetrije i spljostenosti za raspodelu
no2_years = df_year["NO2"]
sb.distplot(no2_years, fit = norm)
plt.xlabel('Koncentracija NO2')
plt.ylabel('Verovatnoća')

print('koef.asimetrije:  %.2f' % skew(df_year['NO2']))
print('koef.spljoštenosti:  %.2f' % kurtosis(df_year['NO2']))

no2_df = data_set[['NO2', 'year']].groupby(["year"]).median().sort_values(by = 'year', ascending = False)
plt.plot(no2_df, label = 'NO2')
plt.yticks(np.arange(30, 70, 5))
plt.xticks(np.arange(2012, 2018, 1))
plt.ylabel('Najučestalije vrednosti NO2')
plt.xlabel('Godine')
plt.legend()


#_______________________________________________________________________________________________________________
#zavisnost medju obelezjima

dt2 = data_set.groupby(by = ['year', 'month']).mean()
dt2

dt3 = data_set.groupby(by = ['month']).mean()
No2 = dt3["NO2"]
So2 = dt3['SO2']
o3 = dt3['O3']
pm2 = dt3['PM2.5']
pm10 = dt3['PM10']
plt.plot(No2, 'r', label = 'NO2')
plt.plot(So2, 'g', label = 'SO2')
plt.plot(o3, 'b', label = 'O3')
plt.plot(pm2, 'y', label = 'PM2.5')
plt.plot(pm10, 'pink', label = 'PM10')
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(0, 300, 50))
plt.ylabel('Prosečna koncentracija NO2, S02, O3, PM2.5, PM10 i CO')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["Jan", "Feb", "Mar", "Ap", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.xlabel('Mesec')
plt.legend()
#zakljucak: obelezje O3 za razliku od ostalih vrednosti svoj minimun ima pocetkom i krajem godine, dok se povecava sredinom godine u letnim mesecima


#koncentracija cestica za 2014. godinu, ova godina je odabrana jer je prva za koju postoje merenja za sve mesece
N_2014 = dt2.loc[2014]['NO2']
S_2014 = dt2.loc[2014]['SO2']
O_2014 = dt2.loc[2014]['O3']
PM2_2014 = dt2.loc[2014]['PM2.5']
P10_2014 = dt2.loc[2014]['PM10']


plt.plot( N_2014, 'r', label = 'N_2014')
plt.plot( S_2014, 'g', label = 'S_2014')
plt.plot( O_2014, 'b', label = 'O_2014')
plt.plot( PM2_2014, 'y', label = 'PM2_2014')
plt.plot( P10_2014, 'pink', label = 'P10_2014')
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(0, 300, 50))
plt.ylabel('Prosečna koncentracija NO2, S02, O3, PM2.5 i PM10 u vazduhu')
plt.xlabel('Mesec')
plt.legend()

#zavisnost izmedju NO2 i PM2.5 za 2014. godinu
#u vecini slucajeva, veca koncentracija NO2 podrazumevala je i vecu koncentraciju PM2.5
plt.scatter(N_2014, PM2_2014)
plt.xlabel('Koncentracija NO2 u 2014. po mesecima')
plt.ylabel('Koncentracija PM2.5 u 2014. po mesecima')

T_2014 = dt2.loc[2014]['TEMP']

#zavisnost izmedju NO2 i O3 za 2014. godinu
#moze se primetiti negativna povezanost gde za vece vrednosti O3 odgovara manja vrednost NO2, povecanjem NO2 u vazduhu smanjuje se O3
plt.scatter(N_2014, O_2014)
plt.xlabel('Koncentracija NO2 u 2014. po mesecima')
plt.ylabel('Koncentracija 03 u 2014. po mesecima')

#_________________________________________________
#zavisnosti izmedju obelezja

plt.scatter(data_set["year"], data_set["NO2"])
plt.xticks(np.arange(2012, 2018, 1))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Godina')
plt.ylabel('Koncentracija NO2')
#opis: u svakoj godini koncentracija NO2 se krece od 0 do priblizno 200 u 2014. godini, a u svim ostalim godinama i preko 200 μg/m³

plt.scatter(data_set["month"], data_set["NO2"])
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Mesec')
plt.ylabel('Koncentracija NO2')
#opis: najmanja koncentracija azot-dioksida je u 7. mesecu


plt.figure(figsize = (20, 15))
plt.scatter(data_set["day"], data_set["NO2"])
plt.xticks(np.arange(1, 32, 1))
plt.yticks(np.arange(0, 300, 10))
plt.xlabel('Dan')
plt.ylabel('Koncentracija NO2')

plt.figure(figsize = (20, 15))
plt.scatter(data_set["hour"], data_set["NO2"])
plt.xticks(np.arange(0, 24, 1))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Sat')
plt.ylabel('Koncentracija NO2')
# opis: primecuje se povecanje koncentracije NO2 u popodnevnim casovima


plt.figure(figsize = (15, 10))
plt.scatter(data_set["PM2.5"], data_set["NO2"])
plt.xticks(np.arange(0, 800, 50))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Koncentracija PM2.5')
plt.ylabel('Koncentracija NO2')
#porast koncentracije PM2.5 ujedno oznacava i porast koncentracije NO2
#primecuju se pojedine tacke koje odstupaju od navedenog, npr. PM2.2 je iznosio oko 500 μg/m³, a NO2 je imao vrednost oko 50 μg/m³.  


plt.figure(figsize = (15, 10))
plt.scatter(data_set["PM10"], data_set["NO2"])
plt.xticks(np.arange(0, 1000, 50))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Koncentracija PM10')
plt.ylabel('Koncentracija NO2')
#porast koncentracije PM10 u vecini slucajeva podrazumeva i porast NO2
#ipak, postoji veliki broj slucajeva koji oznacavaju porast PM10, ali ne i porast NO2


plt.figure(figsize = (15, 10))
plt.scatter(data_set["SO2"], data_set["NO2"])
plt.xticks(np.arange(0, 300, 50))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Koncentracija SO2')
plt.ylabel('Koncentracija NO2')
#nema znacajne povezanosti izmedju koncentracije NO2 i SO2


plt.figure(figsize = (10, 7))
plt.scatter(data_set["CO"], data_set["NO2"])
plt.xticks(np.arange(0, 10000, 1000))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Koncentracija CO')
plt.ylabel('Koncentracija NO2')
#u velikom broju slucajeva, veca koncentracija CO znaci i veca koncentracija NO2 - pozitivna povezanost
#postoje vrednosti koje dokazuju suprotno kada za iste vrednosti CO, vrednosti NO2 su se povecavale

plt.figure(figsize = (10, 10))
plt.scatter(data_set["O3"], data_set["NO2"])
plt.xticks(np.arange(0, 800, 50))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Koncentracija O3')
plt.ylabel('Koncentracija NO2')
#moze se primetiti sto je veca koncentracija O3, to je manja koncentracija NO2 - negativna povezanost
#za odredjeni broj slucajeva za istu koncentraciju O3, vece su bile vrednosti NO2

plt.figure(figsize = (15, 10))
plt.scatter(data_set["TEMP"], data_set["NO2"])
plt.xticks(np.arange(-17, 40, 5))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Temperatura')
plt.ylabel('Koncentracija NO2')
#negativna povezanost izmedju analiziranih obelezja, ali ta povezanost nije znacajna

plt.figure(figsize = (15, 10))
plt.scatter(data_set["PRES"], data_set["NO2"])
plt.xticks(np.arange(980, 1050, 10))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Pritisak')
plt.ylabel('Koncentracija NO2')
#nema znacajne povezanosti izmedju NO2 i Pritiska

plt.figure(figsize = (15, 10))
plt.scatter(data_set["DEWP"], data_set["NO2"])
plt.xticks(np.arange(-30, 35, 15))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Tačka rose')
plt.ylabel('Koncentracija NO2')
#nema znacajne povezanosti izmedju NO2 i Tacke rose

plt.figure(figsize = (15, 10))
plt.scatter(data_set["RAIN"], data_set["NO2"])
plt.xticks(np.arange(0, 47, 5))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Kolicina padavina')
plt.ylabel('Koncentracija NO2')
#postoji minimalna negativna povezanost izmedju NO2 i kolicine padavina

plt.figure(figsize = (15, 10))
plt.scatter(data_set["wd"], data_set["NO2"])
plt.xticks(np.arange(0, 17, 1))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Smer vetra')
plt.ylabel('Koncentracija NO2')


plt.figure(figsize = (15, 10))
plt.scatter(data_set["WSPM"], data_set["NO2"])
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 300, 50))
plt.xlabel('Brzina vetra')
plt.ylabel('Koncentracija NO2')
#sto je brzina vetra manja, to je veca koncentracija NO2 - negativno povezani

#_________________________________________________
#korelacija

corr = data_set.corr()
f = plt.figure(figsize=(12, 9))
sb.heatmap(corr, annot=True);
#opis: moze se primetiti jaka negativna korelacija izmedju temperature i pritiska vazduha, kao i izmedju tacke rose i pritiska vazduha
#velika pozitivna korelacija izmedju obelezja PM2.5 i PM10, odnosno vece vrednosti PM2.5 uticu i na vece vrednosti PM10
#srednje do jaka pozitivna korelacija izmedju NO2 i CO
#ostale korelacije izmedju obelezja nisu toliko izrazene

df_month = pd.DataFrame()
for i in data_set_year.index.unique():
    df_month[i] = dt2.loc[i, 'NO2']
c1=df_month[2014].corr(df_month[2015])
c2=df_month[2014].corr(df_month[2016])
c4=df_month[2014].corr(df_month[2014])

print("korelacija: %.3f" % c1)
print("korelacija: %.3f" % c2)
print("korelacija: %.3f" % c4)

matrica_korelacije = df_month.corr() 
print(matrica_korelacije[2014])

sb.heatmap(matrica_korelacije, annot=True)

plt.figure(figsize = (8, 5))
plt.scatter(data_set["PRES"], data_set["TEMP"])
plt.xticks(np.arange(980, 1050, 10))
plt.yticks(np.arange(-17, 40, 5))
plt.xlabel('Pritisak vazduha')
plt.ylabel('Temperatura vazduha')
plt.grid()

plt.figure(figsize = (8, 5))
plt.scatter(data_set["PM10"], data_set["PM2.5"])
plt.xticks(np.arange(0, 1000, 100))
plt.yticks(np.arange(0, 900, 100))
plt.xlabel('Koncentracija čestice PM2.5')
plt.ylabel('Koncentracija čestice PM10')
plt.grid()

#____________________________________
#stavka 12
#dodato je novo obelezje u bazu koje se odnosi na dan u sedmici, i racunala se vrednost PM2.5 kako se menja tokom sedmice svake godine

data_set_dodatno_obelezje = data_set

data_set_dodatno_obelezje['dayOfWeek'] = data_set_dodatno_obelezje.apply(lambda x: datetime.date(int(x['year']), int(x['month']), int(x['day'])).strftime("%w"), axis=1)

data_set_dodatno_obelezje2013w = data_set_dodatno_obelezje[data_set_dodatno_obelezje['year'] == 2013].groupby('dayOfWeek').mean()
data_set_dodatno_obelezje2013w.plot.line(y=['PM2.5'],title="2013")
plt.xticks([0, 1, 2, 3, 4, 5, 6], ["SU", "MO", "TU", "WE", "TH", "FR", "SA"])

data_set_dodatno_obelezje2014w = data_set_dodatno_obelezje[data_set_dodatno_obelezje['year'] == 2014].groupby('dayOfWeek').mean()
data_set_dodatno_obelezje2014w.plot.line(y=['PM2.5'],title="2014")
plt.xticks([0, 1, 2, 3, 4, 5, 6], ["SU", "MO", "TU", "WE", "TH", "FR", "SA"])

data_set_dodatno_obelezje2015w = data_set_dodatno_obelezje[data_set['year'] == 2015].groupby('dayOfWeek').mean()
data_set_dodatno_obelezje2015w.plot.line(y=['PM2.5'],title="2015")
plt.xticks([0, 1, 2, 3, 4, 5, 6], ["SU", "MO", "TU", "WE", "TH", "FR", "SA"])

data_set_dodatno_obelezje2016w = data_set_dodatno_obelezje[data_set['year'] == 2016].groupby('dayOfWeek').mean()
data_set_dodatno_obelezje2016w.plot.line(y=['PM2.5'],title="2016")
plt.xticks([0, 1, 2, 3, 4, 5, 6], ["SU", "MO", "TU", "WE", "TH", "FR", "SA"])

data_set_dodatno_obelezje2017w = data_set_dodatno_obelezje[data_set['year'] == 2017].groupby('dayOfWeek').mean()
data_set_dodatno_obelezje2017w.plot.line(y=['PM2.5'],title="2017")
plt.xticks([0, 1, 2, 3, 4, 5, 6], ["SU", "MO", "TU", "WE", "TH", "FR", "SA"])

#zakljucak: za sve godine, krajem sedmice, konkretno subotom vrednost PM2.5 je povecana

#_______________________________________
#linearna regresija

#nezavisne promenljive
x = data_set[["year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd"]]

corr = x.corr()
f = plt.figure(figsize=(12, 9))
sb.heatmap(corr.abs(), annot=True);

#zavisna promenljiva
y = data_set["NO2"]

#najjednostavniji model
def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))

#podela na test i trening skup
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
print("koeficijenti: ", first_regression_model.coef_)

#rezultat:
#Mean squared error:  289.17928040042267
#Mean absolute error:  12.614632096537175
#Root mean squared error:  17.005272135441487
#R2 score:  0.7188316688558024
#R2 adjusted score:  0.7187065741014242

#standardizacija obelezja
scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)
x_train_std.head()

# Inicijalizacija
regression_model_std = LinearRegression()

# Obuka modela
regression_model_std.fit(x_train_std, y_train)

# Testiranje
y1_predicted = regression_model_std.predict(x_test_std)

# Evaluacija
model_evaluation(y_test, y1_predicted, x_train_std.shape[0], x_train_std.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
print("koeficijenti: ", regression_model_std.coef_)

#zakljucak: nije doslo do poboljsanja modela

#primena selekcije unazad
import statsmodels.api as sm
X = sm.add_constant(x_train)
model = sm.OLS(y_train, X.astype(float)).fit()
model.summary()

#na osnovu dobijenih rezultata, odbacuje se obelezje PRES cija je p vrednost 0.685

#pravi se novi skup x bez obelezja PRES

x_1 = data_set[["year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "CO", "O3", "TEMP", "RAIN", "DEWP", "WSPM"]]

x1_train, x1_test, y_train, y_test = train_test_split(x_1, y, test_size=0.1, random_state=42)

#standardizacija
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_std = scaler.transform(x1_train)
x1_test_std = scaler.transform(x1_test)
x1_train_std = pd.DataFrame(x1_train_std)
x1_test_std = pd.DataFrame(x1_test_std)
x1_train_std.columns = list(x_1.columns)
x1_test_std.columns = list(x_1.columns)
x1_train_std.head()

second_regression_model = LinearRegression(fit_intercept=True)

# Obuka
second_regression_model.fit(x1_train_std, y_train)

# Testiranje
y1_predicted = second_regression_model.predict(x1_test_std)

# Evaluacija
model_evaluation(y_test, y1_predicted, x1_train_std.shape[0], x1_train_std.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(second_regression_model.coef_)),second_regression_model.coef_)
print("koeficijenti: ", second_regression_model.coef_)

#zakljucak: nije doslo do poboljsanja modela, izbacuje se i obelezje RAIN
x_2 = data_set[["year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "CO", "O3", "TEMP", "DEWP", "WSPM",]]

x2_train, x2_test, y_train, y_test = train_test_split(x_2, y, test_size=0.1, random_state=42)

#standardizacija
scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_std = scaler.transform(x2_train)
x2_test_std = scaler.transform(x2_test)
x2_train_std = pd.DataFrame(x2_train_std)
x2_test_std = pd.DataFrame(x2_test_std)
x2_train_std.columns = list(x_2.columns)
x2_test_std.columns = list(x_2.columns)
x2_train_std.head()

third_regression_model = LinearRegression(fit_intercept=True)

# Obuka
third_regression_model.fit(x2_train_std, y_train)

# Testiranje
y2_predicted = third_regression_model.predict(x2_test_std)

# Evaluacija
model_evaluation(y_test, y2_predicted, x2_train_std.shape[0], x2_train_std.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(third_regression_model.coef_)),third_regression_model.coef_)
print("koeficijenti: ", third_regression_model.coef_)

#zakljucak: nakon selekcije unazad nije doslo do poboljsanja modela
#u nastavku se koristi pocetni skup x 

#upotreba ugradjene funkcije za selekciju unapred

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

# Sequential Forward Selection(sfs)
sfs = SFS(LinearRegression(),
          k_features=12,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)


sfs.fit(x, y)
sfs.k_feature_names_


#2. stepen
poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)

regression_model_inter = LinearRegression()

# Obuka modela
regression_model_inter.fit(x_inter_train, y_train)

# Testiranje
y3_predicted = regression_model_inter.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y3_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
print("koeficijenti: ", regression_model_inter.coef_)

#regularizacija
ridge_model = Ridge(alpha=10)

# Obuka modela
ridge_model.fit(x_inter_train, y_train)

# Testiranje
y4_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y4_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
print("koeficijenti: ", ridge_model.coef_)


#3. stepen
poly = PolynomialFeatures(degree = 3, interaction_only=False, include_bias=False)
x1_inter_train = poly.fit_transform(x_train_std)
x1_inter_test = poly.transform(x_test_std)

regression_model_inter1 = LinearRegression()

# Obuka modela
regression_model_inter1.fit(x1_inter_train, y_train)

# Testiranje
y5_predicted = regression_model_inter1.predict(x1_inter_test)

# Evaluacija
model_evaluation(y_test, y5_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter1.coef_)),regression_model_inter1.coef_)
print("koeficijenti: ", regression_model_inter1.coef_)

#regularizacija
# Inicijalizacija
ridge_model1 = Ridge(alpha=10)

# Obuka modela
ridge_model1.fit(x1_inter_train, y_train)

# Testiranje
y6_predicted = ridge_model1.predict(x1_inter_test)

# Evaluacija
model_evaluation(y_test, y6_predicted, x1_inter_train.shape[0], x1_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model1.coef_)),ridge_model1.coef_)
print("koeficijenti: ", ridge_model1.coef_)

regr_cv = RidgeCV(alphas=[5, 10, 15, 20, 25, 30, 35])
model_cv = regr_cv.fit(x_train_std, y_train)
model_cv.alpha_

# Lasso regresija

# Model initialization
lasso_model = Lasso(alpha=2)

# Fit the data(train the model)
lasso_model.fit(x1_inter_train, y_train)

# Predict
y7_predicted = lasso_model.predict(x1_inter_test)

# Evaluation
model_evaluation(y_test, y7_predicted, x1_inter_train.shape[0], x1_inter_train.shape[1])

plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
print("koeficijenti: ", lasso_model.coef_)
#primenom lasso regularizacione tehnike nije doslo do poboljsanja mera uspesnosti


























