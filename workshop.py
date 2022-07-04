import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import timeit

# 1. Daten
# Daten laden und anschauen
data = pd.read_csv("Aschach_data.csv")
data.head()
# Datum Spalte in index umwandeln und entfernen
data.index = pd.to_datetime(data.date, format = "%Y-%m-%d")
data = data.drop(columns = "date")
# Wie lang ist die Zeitreihe?
jahre = data.index.year.unique().tolist()
print(f"Die Daten beinhalten die Jahre {', '.join([str(x) for x in [*jahre]])}")
# Daten-Struktur
data.head()

# 2. Explorative data analysis (EDA)

# To Do
#Schau dir die Daten mal genau an. Code Beispiele dafür findest du unten.
# Versuche folgende Fragen zu beanworten:
#1. Welche Variablen haben potenziell den größten Einfluss auf wt?
#2. Reichen die vorhandenen Variablen aus um wt über das Jahr gut vorherzusagen, oder gibt es andere Einflussgrößen die wir berechnen könnten?
#3. Brauchen wir Informationen der letzten Tage (lags) um wt vorherzusagen?


# Plotte die Zeitreihe(n) einer oder mehrerer Variablen
# Input
var = ["wt"] # Beispiele für meherere Variablen ["Q", "Ta", "wt"]
# Code
plt.figure(figsize=(15,6))
sns.lineplot(data = data[var])
plt.show()

# Plotte zwei Variablen zusammen in einen Scatterplot
# Inputs
var1 = "wt"
var2 = "GL"
# Code
plt.figure(figsize=(7,7))
sns.scatterplot(x=data[var1], y=data[var2])
plt.show()

# Plotte wt vs. lagged variables
# lagged variable = Werte der vorhergehenden Zeitschritte, z.B. Qlag1 ist der Abfluss vom vorherigen Tag
# Inputs
var = "Ta_max"
lag = 8
# Code
lagged_var = pd.concat([pd.Series([np.NaN for i in range(lag+1)]), data[var].iloc[:-(lag+1)]])
lagged_var.index = data[var].index
lagged_var.name = f"{var}_lag{lag}"
lagged_var.head()
plt.figure(figsize=(7,7))
sns.scatterplot(x=data.wt, y=lagged_var)
plt.show()

# Plotte die Korrelation aller Variablen
# Hier gibt es nichts zum Anpassen --> mach dir ein Bild von den (linearen-) Zusammenhängen zwischen den Variablen
corr = data.corr()
plt.figure(figsize=(9,6))
sns.heatmap(corr, cmap=sns.diverging_palette(230, 20, as_cmap=True), vmin=-1, vmax=1, annot=True)
plt.show()

# 3. Daten pre-processing / feature engineering
# Um die relevanten Informationen für die Vorhersage zu verwenden, werden wir neue Features (Variablen) erzeugen!

# 3.1 Lag Variablen
def create_lags(df, columns, lags):
    """
    Erstellt data frame mit zusätzlich lagged Versionen ausgewählter Spalten des data frames.
    Arguments:
        df: data frame
        columns: Liste mit strings der Spaltennamen
        lags: integer, Anzahl der lags
    Return:
        Einen data frame bestehend aus dem Input data frame mit zusätzlichen Spalten für die lagged Variablen.
    """
    lagged_cols = []
    for var in columns:
        for i in range(lags):
            lagged_var = pd.concat([pd.Series([np.NaN for lag in range(i+1)]), df[var].iloc[:-(i+1)]])
            lagged_var.index = df[var].index
            lagged_var.name = f"{var}_lag{i+1}"
            lagged_cols.append(lagged_var)
    var_lags = pd.concat(lagged_cols, axis=1)
    return pd.concat([df, var_lags], axis=1)

### To do:
# Wähle eine Anzahl an lags (lag=1 bedeutet, dass die Variablen des Vortages zusätzlich verwendet werden).


# Erzeuge lag Variablen
# Inputs
n_lags = 3
# Code
lagged_cols = [col for col in data.columns if col != "wt"]
data_pp = create_lags(df=data, columns=lagged_cols, lags=n_lags)
# Zeilen mit NaNs entfernen
data_pp.dropna(inplace=True)

# 3.2 Zeit-Information
# Problem: Zeit Variablen sind zyklisch --> 1. Tag und 365. Tag im Jahr sind nahezu ident.
# Wie können wir aus den zyklischen Variablen der Zeit (Tage des Jahres, Monate, Tage im Monat) eine kontinuierliche Zeit-Variable erzeugen?
# -> Sinus/Consinus Transformation
data_pp['sin_time'] = np.sin(2*np.pi*data_pp.index.dayofyear/data_pp.index.dayofyear.max())
data_pp['cos_time'] = np.cos(2*np.pi*data_pp.index.dayofyear/data_pp.index.dayofyear.max())

# Plotte die Zeitreihe(n) einer oder mehrerer Variablen
var = ["sin_time", "cos_time"]
sns.set_style("whitegrid")
plt.figure(figsize=(15,6))
sns.lineplot(data = data_pp[var])
plt.show()

data_pp.head()

# 4. Data Splits (+Präsentation)

# Train/Test split
test_period = [2013, 2014, 2015]
train_data = data_pp.loc[~data_pp.index.year.isin(test_period)]
test_data = data_pp.loc[data_pp.index.year.isin(test_period)]

print(f"Training samples: {train_data.shape[0]} Tage\nTest samples: {test_data.shape[0]} Tage")

# Aufspalten in Inputs (x) und Outputs (y)
x_train = train_data.loc[:, train_data.columns != "wt"]
y_train = train_data.wt
x_test = test_data.loc[:, test_data.columns != "wt"]
y_test = test_data.wt

# Plotte Training & Test Split
sns.set_style("whitegrid")
plt.figure(figsize=(15,7))
plt.title("Mittlere tägliche Wassertemperatur der Aschach")
plt.plot(train_data.index, train_data.wt, color='#379BDB', label='Training')
plt.plot(test_data.index, test_data.wt, color='#fc7d0b', label='Test')
plt.xlabel('')
plt.ylabel('Wassertemperatur [°C]')
plt.legend()
plt.show(block=False)

# 5. Baseline Modell

# Lineares Modell mit 10-facher CV
start = timeit.default_timer()
x_baseline = x_train.filter(items=data.columns)
baseline_model = linear_model.LinearRegression()
baseline_scores = cross_val_score(baseline_model, x_train.filter(items=data.columns), y_train,
                                  scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
end = timeit.default_timer()

# Model performance
print(f"RMSE: {-np.mean(baseline_scores):.3f} ({np.std(-baseline_scores):.3f})")
#print(end - start)
print(f"Laufzeit: {end - start:.2f} Sekunden")


# Definiere Gütemaße
def RMSE(x, y):
    return round(((x - y) ** 2).mean() ** .5, 3)
def MAE(x, y):
    return round((x - y).abs().mean(), 3)
def print_results(x, y, name):
    print(f"{name} RMSE = {RMSE(x, y)} °C\n{name} MAE = {MAE(x, y)} °C")

# Erzeuge Baseline Vorhersage der Test Daten
baseline_model.fit(x_train.filter(items=data.columns), y_train)
baseline_prediction = baseline_model.predict(x_test.filter(items=data.columns))
baseline_prediction_df = pd.DataFrame({"prediction": baseline_prediction})
baseline_prediction_df.index = test_data.index

# Plotte Baseline Vorhersage
sns.set_style("whitegrid")
plt.figure(figsize=(15,7))
plt.title("Mittlere tägliche Wassertemperatur der Aschach")
plt.plot(test_data.index, test_data.wt, color= '#379BDB', label='Beobachtung')
plt.plot(baseline_prediction_df.index, baseline_prediction_df.prediction, color='#fc7d0b', label='Baseline Modell')
plt.xlabel('')
plt.ylabel('Wassertemperatur [°C]')
plt.legend()
plt.show(block=False)

print_results(test_data.wt, baseline_prediction, "Baseline")

## 6. Machine Learning Modell (+Präsentation)

# 6.1 Hyperparameter Auswahl

#gewähltes Modell: Random Forest

# To do:
# Probiere unterschiedliche Hyperparameter aus und versuche die beste Kombination zu finden!
# Genauere Beschreibung der Parameter von Random Forest findest du auf https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# Hyperparameter
hyperpara = {"n_estimators": 100,    # default=100, Anzahl der Regressionsbäume
             "max_depth": 4,      # default=None, Tiefe des Baums, bei None werden Bäume so lange expandiert bis "min_samples_split" erreicht ist
             "min_samples_split": 2, # default=2, Minimum samples um eine node zu spalten (muss > 1 sein)
             "min_samples_leaf": 1,  # default=1, Mindestanzahl an samples für einen node
             "max_features": 1.0,    # default=1.0, Anzahl an features für split, default=1.0 (100% aller Variablen werden verwendet)
            }

# Cross Validierung des Random Forest Modells mit den gewählten Hyperparametern
start = timeit.default_timer() # timer
model = RandomForestRegressor(**hyperpara,
                              criterion="squared_error",
                              random_state=42) # Modell mit Hyperparameter
n_scores = cross_val_score(model, x_train, y_train, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
end = timeit.default_timer()

# Model performance
print("Random Forest scores")
print(f"RMSE: {-np.mean(n_scores):.3f} ({np.std(-n_scores):.3f}°C), verbesserung zur Baseline: {-np.mean(baseline_scores) + np.mean(n_scores):.3f}°C")
print(f"Laufzeit: {end - start:.2f} Sekunden")

# Plot training Vorhersage
sns.set_style("whitegrid")
plt.figure(figsize=(18,7))
plt.title("Mittlere tägliche Wassertemperatur der Aschach")

# Training Daten
plt.plot(train_data.index, train_data.wt, color= '#379BDB', label='Beobachtung')

# Random Forest
model.fit(x_train, y_train) # Model fit
train_prediction = model.predict(x_train)
train_prediction_df = pd.DataFrame({"prediction": train_prediction})
train_prediction_df.index = train_data.index
plt.plot(train_prediction_df.index, train_prediction_df.prediction, color='#18d30c', label='Random Forest')
plt.xlabel('')
plt.ylabel('Wassertemperatur [°C]')
plt.legend()
plt.show(block=False)

# 6.2 Modellgüte mit den Test Daten bestimmen
# Model fit und Vorhersage der Test Daten
model.fit(x_train, y_train)
prediction = model.predict(x_test)
prediction_df = pd.DataFrame({"prediction": prediction})
prediction_df.index = test_data.index
print_results(test_data.wt, prediction, "Random Forest")

# Verbesserung im Vergleich zum Baseline Modell
rmse_red = RMSE(test_data.wt, baseline_prediction) - RMSE(test_data.wt, prediction)
mae_red = MAE(test_data.wt, baseline_prediction) - MAE(test_data.wt, prediction)
print(f"RMSE Verbesserung im Vergleich zum Baseline Modell: {rmse_red:.3f} °C")
print(f"MSE Verbesserung im Vergleich zum Baseline Modell: {mae_red:.3f} °C")

# Plot der Test Vorhersagen für ein Testjahr: Beobachtungen, Baseline & Random Forest
jahr = 2014
sns.set_style("whitegrid")
plt.figure(figsize=(18,7))
plt.title("Mittlere tägliche Wassertemperatur der Aschach")

# Test Daten
test_plot = test_data[test_data.index.year == jahr]
plt.plot(test_plot.index, test_plot.wt, color= '#379BDB', label='Beobachtung')
# Baseline Modell
base_plot = baseline_prediction_df[baseline_prediction_df.index.year == jahr]
plt.plot(base_plot.index, base_plot.prediction, color='#fc7d0b', label='Baseline Modell')
# Random Forest
pred_plot = prediction_df[prediction_df.index.year == jahr]
plt.plot(pred_plot.index, pred_plot.prediction, color='#18d30c', label='Random Forest')
plt.xlabel('')
plt.ylabel('Wassertemperatur [°C]')
plt.legend()
plt.show(block=False)

## 7. Zweites ML Modell

# Wähle eines zweites ML Modell und wende es auf die Daten an. Alles ist erlaubt.
# Finde anhand der Beschreibungen ein passendes Modell für dieses (regressions Problem).
# Nutze dazu die Liste an Modellen in scikit-learn: https://scikit-learn.org/stable/supervised_learning.html


# Als Hilfestellung findest du nachfolgend ein Template:

# Cross Validierung eines Modells
start = timeit.default_timer() # timer

# ----------------------------------------
# Hier muss das Modell (und gegebenenfalls die Hyperparameter) definiert werden
model2 = ...
# ----------------------------------------

# Cross Validierung des Modells
n_scores = cross_val_score(model2, x_train, y_train, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
end = timeit.default_timer()

# Model performance
print(f"RMSE: {-np.mean(n_scores):.3f} ({np.std(-n_scores):.3f}), verbesserung zur Baseline: {-np.mean(baseline_scores) + np.mean(n_scores):.3f}°C"))
print(f"Laufzeit: {end - start:.2f} Sekunden")

# Modell fit und Vorhersage der Test Daten
model2.fit(x_train, y_train) # Model fit
prediction2 = model2.predict(x_test)
prediction2_df = pd.DataFrame({"prediction": prediction2})
prediction2_df.index = test_data.index
print_results(test_data.wt, prediction2, "Alternativmodell")

# Verbesserung im Vergleich zum Baseline Modell
rmse_red = RMSE(test_data.wt, baseline_prediction) - RMSE(test_data.wt, prediction2)
mae_red = MAE(test_data.wt, baseline_prediction) - MAE(test_data.wt, prediction2)
print(f"RMSE Verbesserung im Vergleich zur Baseline: {rmse_red:.3f} °C")
print(f"MSE Verbesserung im Vergleich zur Baseline: {mae_red:.3f} °C")

# Verbesserung im Vergleich zum Random Forest Modell
rmse_red = RMSE(test_data.wt, prediction) - RMSE(test_data.wt, prediction2)
mae_red = MAE(test_data.wt, prediction) - MAE(test_data.wt, prediction2)
print(f"RMSE Verbesserung im Vergleich zum Random Forest: {rmse_red:.3f} °C")
print(f"MSE Verbesserung im Vergleich zum Random Forest: {mae_red:.3f} °C")

# 8. Evaluierung der Ergebnisse
# Gruppendiskussion:
# 1. Welches Modell und welche Hyperparameter liefern die besten Ergebnisse?
# 2. Wie könnten wir unsere Modellierung verbessern?