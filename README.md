# CODSOFT Machine Learning Projects

Five standalone ML scripts completed as part of the CODSOFT internship. Each script handles its own data acquisition and runs end-to-end from raw CSV to model evaluation.

## Projects

| # | Script | Dataset | Algorithm | Task |
|---|--------|---------|-----------|------|
| 1 | `task1_titanic_survival.py` | Titanic (auto-downloaded) | Logistic Regression | Binary classification |
| 2 | `task2_movie_rating_prediction.py` | IMDB Indian Movies (Kaggle) | Random Forest Regressor | Regression |
| 3 | `task3_iris_classification.py` | Iris (auto-downloaded) | Random Forest Classifier | Multi-class classification |
| 4 | `task4_sales_prediction.py` | Advertising spend (auto-downloaded) | Linear Regression | Regression |
| 5 | `task5_credit_card_fraud.py` | Credit Card Fraud (Kaggle) | Random Forest + SMOTE | Imbalanced classification |

## Setup

```bash
git clone https://github.com/Vighnesh1045/CODSOFT
cd CODSOFT
pip install -r requirements.txt
```

Tasks 1, 3, and 4 download their datasets automatically on first run.

Tasks 2 and 5 require Kaggle datasets — download them before running:

**Task 2 — IMDB Indian Movies:**
```bash
pip install kaggle
kaggle datasets download -d PromptCloudHQ/imdb-indian-movies-dataset
unzip imdb-indian-movies-dataset.zip
```

**Task 5 — Credit Card Fraud:**
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

## Running

```bash
python task1_titanic_survival.py
python task2_movie_rating_prediction.py   # requires movies.csv
python task3_iris_classification.py
python task4_sales_prediction.py
python task5_credit_card_fraud.py         # requires creditcard.csv
```

## License

MIT — see [LICENSE](LICENSE)
