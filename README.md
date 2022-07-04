<img align="right" width="200" src="https://github.com/MoritzFeigl/ML_workshop/blob/main/Abb/baseflow_txt1.png">

# Machine Learning (in der Hydrologie)  
Workshop 5. Juli, 2022

Dieses Repository beinhaltet alle Skripte und die zugehörige Präsentation für den Workshop. Nach dem Download dieses Repository können die notwendigen python Packages mit dem bash script "workshop_setup.sh" in Linux installiert werden, oder mit folgenden Befehlen im Anaconda Prompt in allen anderen Betriebssystemen:

```
conda create --name workshop python=3.9
conda activate workshop
conda install -c anaconda jupyter
conda install -c anaconda pandas
conda install -c conda-forge matplotlib
conda install -c anaconda scikit-learn
conda install -c anaconda seaborn
jupyter-notebook
```

Wobei der letzte Befehl "juypter-notebook" die jupyter Anwendung in dem neu geschaffenen workshop environment startet.

## Inhalt dieses Repository
- `workshop.ipynb` workshop jupyter-notebook
- `workshop.py` der workshop.ipynb als .py scipts (optional falls jupyter nicht vorhanden)
- `workshop.ppt` workshop slides
- `Aschach_data.csv` Daten der Aschach für die Jahre 2007-20015
- `Abb/` Abbildungen
