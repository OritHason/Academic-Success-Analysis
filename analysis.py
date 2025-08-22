"""
Analysis Script for Student Habits and Academic Performance
---------------------------------------------------------

This script replicates the analyses and figures presented in the supplied
slides.  It loads the ``student_habits_performance.csv`` dataset, computes
summary statistics, performs hypothesis tests, produces plots, and builds
classification models to predict whether a student will obtain a high exam
score.  Two types of classifiers are implemented: a logistic regression
and a random forest.  All results are printed to standard output and
figures are written to the ``figures/`` directory.

Usage
-----
From the root of the git repository, run::

    python analysis.py

The script requires the following Python packages: ``pandas``, ``numpy``,
``matplotlib``, ``seaborn``, ``scipy``, ``statsmodels`` and
``scikit-learn``.  These can be installed with::

    pip install -r requirements.txt

Author: Automated by ChatGPT for reproducibility.
"""

import os
import warnings
from pathlib import Path
from typing import Text, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

# Silence warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def describe_exam_scores(data: pd.DataFrame) -> None:
    """Compute and print summary statistics for exam scores along with
    normality tests and a confidence interval for the mean.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing an ``exam_score`` column.
    """
    scores = data["exam_score"].dropna()
    n = len(scores)
    mean = scores.mean()
    std = scores.std(ddof=1)
    jb_stat, jb_p = stats.jarque_bera(scores)
    # 95% confidence interval for the mean under normality assumption
    alpha = 0.05
    se = std / np.sqrt(n)
    ci_low = mean - stats.t.ppf(1 - alpha / 2, df=n - 1) * se
    ci_high = mean + stats.t.ppf(1 - alpha / 2, df=n - 1) * se

    print("Summary statistics for exam scores:")
    print(f"  Sample size (n): {n}")
    print(f"  Mean (μ̂): {mean:.3f}")
    print(f"  Standard deviation (σ̂): {std:.3f}")
    print(f"  Jarque–Bera test statistic: {jb_stat:.3f}")
    print(f"  Jarque–Bera p-value: {jb_p:.5f}")
    print(
        f"  95% CI for the mean: [{ci_low:.3f}, {ci_high:.3f}]\n"
    )

    # Hypothesis test: H0: μ = 60 vs H1: μ > 60
    t_stat, p_val = stats.ttest_1samp(scores, popmean=60)
    # Since we test one-sided H1: μ > 60, adjust p-value
    p_val_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    print("One–sample t-test for H0: μ = 60 vs H1: μ > 60:")
    print(f"  t statistic: {t_stat:.3f}")
    print(f"  p-value (one-sided): {p_val_one_sided:.5e}\n")


def plot_exam_distribution(data: pd.DataFrame, out_dir: Path) -> None:
    """Plot histogram of exam scores with a fitted normal distribution.

    The resulting figure is saved to `out_dir`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing an ``exam_score`` column.
    out_dir : Path
        Directory where the plot will be saved.
    """
    ensure_directory(out_dir)
    scores = data["exam_score"].dropna()
    mean = scores.mean()
    std = scores.std(ddof=1)

    # Generate normal distribution values for overlay
    x_vals = np.linspace(scores.min(), scores.max(), 200)
    normal_pdf = stats.norm.pdf(x_vals, mean, std)

    plt.figure(figsize=(8, 5))
    # Histogram
    sns.histplot(scores, bins=30, stat="density", color="#A0C4FF", edgecolor="black")
    # Normal curve overlay
    plt.plot(
        x_vals,
        normal_pdf,
        color="red",
        linewidth=2,
        label=f"Normal dist (μ={mean:.1f}, σ={std:.1f})",
    )
    # Mean line
    plt.axvline(mean, color="blue", linestyle="--", linewidth=2, label=f"Mean = {mean:.1f}")
    plt.title("Distribution of Exam Scores")
    plt.xlabel("Exam Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "exam_score_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Exam score distribution plot saved to {fig_path}")


def distributions_grid(data: pd.DataFrame, out_dir: Path) -> None:
    """Plot distributions grid of the numeric parameters.

    The resulting figure is saved to `out_dir`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing an ``exam_score`` column.
    out_dir : Path
        Directory where the plot will be saved.
    """
    ensure_directory(out_dir)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols[: len(axes)]):
        sns.histplot(data[col], ax=axes[i], kde=True)
        axes[i].set_title(f'{col}', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Remove any unused axes
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = Path(out_dir) / "distributions_grid.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Distributions grid saved to {path}")


def scatter_grid_vs_score(data: pd.DataFrame, out_dir: Path, color_by: Text) -> None:
    """Plot scatter grid of the numeric parameters compare to the score.

    The resulting figure is saved to `out_dir`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing an ``exam_score`` column.
    out_dir : Path
        Directory where the plot will be saved.
    color_by: Text
        Field to color the points
    """
    ensure_directory(out_dir)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "exam_score"]
    cols = 3
    rows = int(np.ceil(len(numeric_cols) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.scatterplot(data=data, x=col, y='exam_score', hue=color_by, ax=axes[i], alpha=0.6)
        axes[i].set_title(f'{col} vs exam_score')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('exam_score')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Scatter Grid", y=1.02)
    plt.tight_layout()
    path = Path(out_dir) / "scatter_grid_vs_score.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Scatter grid vs score saved to {path}")


def correlation_matrix(data: pd.DataFrame, out_dir: Path):
    """Plot correlation matrix for the data.

    The resulting figure is saved to `out_dir`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing an ``exam_score`` column.
    out_dir : Path
        Directory where the plot will be saved.
    """
    ensure_directory(out_dir)
    plt.figure(figsize=(8, 6))
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    path = Path(out_dir) / "correlation_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Correlation matrix saved to {path}")


def correlation_analysis(data: pd.DataFrame, out_dir: Path) -> None:
    """Compute correlation between study hours and exam score and plot the relationship.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing ``study_hours_per_day`` and ``exam_score`` columns.
    out_dir : Path
        Directory where the plot will be saved.
    """
    ensure_directory(out_dir)
    x = data["study_hours_per_day"].dropna()
    y = data.loc[x.index, "exam_score"]
    # Pearson correlation
    r, p_val = stats.pearsonr(x, y)
    print("Correlation between study hours and exam score:")
    print(f"  Pearson r: {r:.3f}")
    print(f"  p-value: {p_val:.3e}\n")

    # Scatter plot with regression line
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x, y=y, color="#FB8B24", alpha=0.6, edgecolor=None)
    # Fit simple linear regression for line of best fit
    slope, intercept = np.polyfit(x, y, 1)
    plt.plot(x, slope * x + intercept, color="red", linewidth=2)
    plt.title("Exam Score vs Study Hours")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Exam Score")
    plt.tight_layout()
    fig_path = Path(out_dir) / "exam_vs_study_hours.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Exam score vs study hours plot saved to {fig_path}\n")


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix for classification.

    This function encodes categorical variables using one-hot encoding and drops
    non-predictive columns such as the student identifier and the exam score.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        Features X and target y.
    """
    # Copy to avoid modifying the original
    data = df.copy()

    # Create binary target: 1 if exam_score >= 70, else 0
    data["success"] = (data["exam_score"] >= 70).astype(int)

    # Drop identifier and target columns from features
    X = data.drop(columns=["student_id", "exam_score", "success"])
    y = data["success"]

    # Identify categorical and numerical columns
    categorical_cols = [
        "gender",
        "part_time_job",
        "diet_quality",
        "parental_education_level",
        "internet_quality",
        "extracurricular_participation",
    ]

    # One-hot encode categorical variables, drop first level to avoid multicollinearity
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X_encoded, y


def logistic_model_statsmodels(X: pd.DataFrame, y: pd.Series) -> None:
    """Fit a logistic regression model using statsmodels and print the summary.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target variable.
    """
    # Add constant term for intercept and ensure numeric types
    X_sm = sm.add_constant(X)
    X_sm = X_sm.astype(float)
    y_numeric = y.astype(float)
    logit_model = sm.Logit(y_numeric, X_sm)
    result = logit_model.fit(disp=False)
    print("Logistic regression (statsmodels) summary:")
    print(result.summary2().as_text())

    # Likelihood ratio test comparing the full model to an intercept-only model
    ll_full = result.llf
    # Fit a null model with only the intercept
    null_model = sm.Logit(y_numeric, np.ones((len(y_numeric), 1))).fit(disp=False)
    ll_null = null_model.llf
    lr_stat = -2 * (ll_null - ll_full)
    df_diff = X.shape[1]  # degrees of freedom difference
    p_value = stats.chi2.sf(lr_stat, df=df_diff)
    print("\nLikelihood ratio test (full model vs null model):")
    print(f"  LR statistic: {lr_stat:.3f}")
    print(f"  Degrees of freedom: {df_diff}")
    print(f"  p-value: {p_value:.3e}")
    if 'study_hours_per_day' in X.columns:
        coef = result.params['study_hours_per_day']
        print(
            f"\nCoefficient for study_hours_per_day: {coef:.3f} (log-odds); "
            f"exp(coef) ≈ {np.exp(coef):.1f}"
        )
    print()


def _plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    title: str,
    out_path: Path,
    normalize: bool = False,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        cm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        square=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def classification_models(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    """Train logistic regression and random forest classifiers and evaluate them.

    Saves ROC curves and confusion matrices, and prints classification metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target variable.
    out_dir : Path
        Directory where plots will be saved.
    """
    ensure_directory(out_dir)
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize numerical features for logistic regression
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Logistic Regression (scikit-learn)
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(X_train_scaled, y_train)
    y_pred_lr = lr_clf.predict(X_test_scaled)
    y_proba_lr = lr_clf.predict_proba(X_test_scaled)[:, 1]

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    # For AUC, use predicted probabilities (mean vote of trees)
    if hasattr(rf_clf, "predict_proba"):
        y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]
    else:
        dec = rf_clf.decision_function(X_test)
        if dec.max() == dec.min():
            y_proba_rf = np.full_like(dec, fill_value=0.5, dtype=float)
        else:
            y_proba_rf = (dec - dec.min()) / (dec.max() - dec.min())

    # Compute evaluation metrics
    def print_metrics(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"{name} Classification Metrics:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Confusion matrix:\n{cm}")
        print(classification_report(y_true, y_pred, digits=3))

    print_metrics("Logistic Regression (sklearn)", y_test, y_pred_lr)
    print_metrics("Random Forest", y_test, y_pred_rf)

    # Confusion matrices (saved as figures)
    class_names = ["Fail (0)", "Pass (1)"]
    _plot_confusion_matrix(
        y_test, y_pred_lr, class_names,
        title="Logistic Regression – Confusion Matrix",
        out_path=Path(out_dir) / "confusion_matrix_lr.png",
        normalize=False,
    )
    _plot_confusion_matrix(
        y_test, y_pred_lr, class_names,
        title="Logistic Regression – Confusion Matrix (Normalized)",
        out_path=Path(out_dir) / "confusion_matrix_lr_normalized.png",
        normalize=True,
    )
    _plot_confusion_matrix(
        y_test, y_pred_rf, class_names,
        title="Random Forest – Confusion Matrix",
        out_path=Path(out_dir) / "confusion_matrix_rf.png",
        normalize=False,
    )
    _plot_confusion_matrix(
        y_test, y_pred_rf, class_names,
        title="Random Forest – Confusion Matrix (Normalized)",
        out_path=Path(out_dir) / "confusion_matrix_rf_normalized.png",
        normalize=True,
    )
    print(
        f"Confusion matrices saved to {out_dir}/confusion_matrix_lr.png, "
        f"{out_dir}/confusion_matrix_lr_normalized.png, "
        f"{out_dir}/confusion_matrix_rf.png, "
        f"{out_dir}/confusion_matrix_rf_normalized.png"
    )

    # ROC curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, color="blue", lw=2, label=f"Logistic Regression (AUC = {roc_auc_lr:.3f})")
    plt.plot(fpr_rf, tpr_rf, color="green", lw=2, label=f"Random Forest (AUC = {roc_auc_rf:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Classification Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = Path(out_dir) / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC curves plot saved to {roc_path}\n")

    # Baseline accuracy: always predict the most frequent class
    majority_class = y.value_counts().idxmax()
    baseline_acc = (y == majority_class).mean()
    print(f"Baseline accuracy (predicting all as {majority_class}): {baseline_acc:.3f}\n")

    # Cross‑validation for logistic regression accuracy and AUC
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Identify numeric columns (for scaling) and the remainder (for passthrough)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)],
        remainder="passthrough",
    )
    cv_clf = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(cv_clf, X, y, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(cv_clf, X, y, cv=cv, scoring="roc_auc")
    print("Cross‑validated performance for logistic regression:")
    print(f"  Mean accuracy over 5 folds: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  Mean AUC over 5 folds: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}\n")


def moderation_plot(data: pd.DataFrame, usage_col: str, low_threshold: float, high_threshold: float, title: str, out_path: Path) -> None:
    """Generate a moderation plot showing how the relationship between study hours
    and exam score differs for low- and high-usage groups.

    The function splits the data into two groups based on the specified
    thresholds for a moderating variable (such as ``netflix_hours`` or
    ``social_media_hours``), plots scatter points for each group, fits a
    separate linear regression line to each group, and annotates the plot
    with the Pearson correlation coefficient for each group.  The result
    is saved to the provided file path.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing ``study_hours_per_day``, ``exam_score`` and the
        moderator column.
    usage_col : str
        Name of the moderating variable column.
    low_threshold : float
        Upper bound for the low-usage group (strict inequality).
    high_threshold : float
        Lower bound for the high-usage group (strict inequality).
    title : str
        Title for the plot.
    out_path : Path
        Path where the figure will be saved.
    """
    ensure_directory(out_path.parent)
    # Define low and high usage groups
    low_group = data[data[usage_col] < low_threshold]
    high_group = data[data[usage_col] > high_threshold]

    # Compute Pearson correlations
    def safe_corr(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return float('nan')
        return stats.pearsonr(group['study_hours_per_day'], group['exam_score'])[0]
    r_low = safe_corr(low_group)
    r_high = safe_corr(high_group)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='study_hours_per_day',
        y='exam_score',
        data=low_group,
        label=f"Low {usage_col.replace('_hours','').capitalize()} (<{low_threshold} h/day) (r={r_low:.2f})",
        color='#1f77b4',
        alpha=0.6,
    )
    sns.scatterplot(
        x='study_hours_per_day',
        y='exam_score',
        data=high_group,
        label=f"High {usage_col.replace('_hours','').capitalize()} (>{high_threshold} h/day) (r={r_high:.2f})",
        color='#ff7f0e',
        alpha=0.6,
    )

    # Regression lines for each group
    for group, color in [(low_group, '#1f77b4'), (high_group, '#ff7f0e')]:
        if len(group) >= 2:
            x = group['study_hours_per_day']
            y = group['exam_score']
            slope, intercept = np.polyfit(x, y, 1)
            x_vals = np.linspace(x.min(), x.max(), 100)
            y_vals = slope * x_vals + intercept
            plt.plot(x_vals, y_vals, color=color, linestyle='--')

    plt.title(title)
    plt.xlabel('Study Hours per Day')
    plt.ylabel('Exam Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Moderation plot saved to {out_path}")


def main() -> None:
    # Set up directories
    project_root = Path(__file__).resolve().parent
    figures_dir = project_root / "figures"
    ensure_directory(figures_dir)

    # Load dataset (assuming it's in the project root)
    data_path = project_root / "student_habits_performance.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)

    # Descriptive statistics and hypothesis tests
    describe_exam_scores(df)

    # Plots
    plot_exam_distribution(df, figures_dir)
    distributions_grid(df, figures_dir)
    correlation_matrix(df, figures_dir)
    scatter_grid_vs_score(df, figures_dir, "gender")
    correlation_analysis(df, figures_dir)

    # Prepare features and target for classification models
    X, y = prepare_features(df)

    # Fit logistic regression using statsmodels for interpretability
    logistic_model_statsmodels(X, y)

    # Train classification models and evaluate (now with CM figures)
    classification_models(X, y, figures_dir)

    # Moderation analysis plots
    # Netflix usage: low <1 hour/day, high >3 hours/day
    moderation_plot(
        df,
        usage_col='netflix_hours',
        low_threshold=1.0,
        high_threshold=3.0,
        title='Moderation by Netflix Usage',
        out_path=figures_dir / 'moderation_netflix.png',
    )
    # Social media usage: low <1 hour/day, high >4 hours/day
    moderation_plot(
        df,
        usage_col='social_media_hours',
        low_threshold=1.0,
        high_threshold=4.0,
        title='Moderation by Social Media Usage',
        out_path=figures_dir / 'moderation_social_media.png',
    )


if __name__ == "__main__":
    main()
