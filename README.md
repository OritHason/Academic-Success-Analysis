# Academic Success Analysis

This repository contains a reproducible analysis of the
`student_habits_performance.csv` dataset.  The data set describes 1,000
students and records a variety of lifestyle factors (study habits,
media consumption, sleep, exercise, parental background, etc.) along
with each student’s exam score.  The purpose of this project is to
explore the distribution of exam scores, understand relationships
between study habits and performance, and build predictive models for
classifying students into high‐ and low‐performing groups.

## Contents

* `student_habits_performance.csv` – the raw data containing 1,000
  observations and 16 variables.
* `analysis.py` – a Python script that reproduces all of the
  descriptive statistics, hypothesis tests, figures and classification
  models shown in the supplied slides.
* `requirements.txt` – a list of Python dependencies needed to run
  the analysis script.
* `figures/` – this directory is created by the analysis script to
  store generated plots (histograms, scatterplots and ROC curves).

## Running the analysis

1. **Install dependencies** – Create a virtual environment (optional)
   and install the required packages:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the script** – From the root of the repository, execute
   `analysis.py`:

   ```bash
   python analysis.py
   ```

   The script will print summary statistics and model results to the
   console.  It also writes the following figures to the `figures/`
   directory:

   * `exam_score_distribution.png` – a histogram of exam scores with
     an overlaid normal distribution and mean marker.
   * `exam_vs_study_hours.png` – a scatter plot of exam score versus
     study hours per day with a fitted regression line.
   * `roc_curves.png` – ROC curves comparing logistic regression and
     random forest classifiers.

## Analysis overview

The analyses performed by `analysis.py` include:

* **Distribution of exam scores** – summary statistics (mean,
  standard deviation), a Jarque–Bera test for normality, and a
  one–sample *t*‑test to evaluate whether the mean exam score is
  significantly greater than 60.  A 95 % confidence interval for
  the mean is also reported.

* **Relationship between study hours and exam performance** – the
  Pearson correlation coefficient between study hours per day and
  exam score is computed, and a scatter plot with a regression line
  is produced.

* **Classification modelling** – a binary outcome variable called
  `success` is defined as 1 if a student’s exam score is at least 70
  (approximately the median) and 0 otherwise.  Two classifiers are
  trained and evaluated:

  1. **Logistic regression** – fitted using both the
     `statsmodels` library (to obtain parameter estimates and
     p‑values) and `scikit‑learn` (for prediction on a held‑out test set).  Continuous
     predictors are standardised.  Accuracy, confusion matrix and
     the receiver–operating characteristic (ROC) curve are reported.

  2. **Random forest** – an ensemble classifier consisting of 200
     decision trees.  It uses the unstandardised features directly.  The
     same evaluation metrics (accuracy, confusion matrix and ROC/AUC)
     are computed.

These analyses mirror those presented in the supplied slides.  You
should be able to run the script on any modern Python installation and
obtain comparable results.