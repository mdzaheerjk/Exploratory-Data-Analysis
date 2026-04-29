# 🧭 Exploratory Data Analysis (EDA) — Complete Notes
### From Absolute Beginner → Advanced ML/DL/GenAI/Agentic AI
#### Dataset Used Throughout: **Titanic Passenger Survival Dataset**

---

## 📌 Table of Contents

1. [What is EDA?](#1-what-is-eda)
2. [Why EDA Matters in AIML Jobs](#2-why-eda-matters-in-aiml-jobs)
3. [The Titanic Dataset — Overview](#3-the-titanic-dataset--overview)
4. [Level 1 — Beginner EDA](#4-level-1--beginner-eda)
5. [Level 2 — Intermediate EDA](#5-level-2--intermediate-eda)
6. [Level 3 — Advanced EDA](#6-level-3--advanced-eda)
7. [Level 4 — EDA for ML/DL Pipelines](#7-level-4--eda-for-mldl-pipelines)
8. [Level 5 — EDA for Generative AI](#8-level-5--eda-for-generative-ai)
9. [Level 6 — EDA for Agentic AI Systems](#9-level-6--eda-for-agentic-ai-systems)
10. [EDA Checklist for AIML Jobs](#10-eda-checklist-for-aiml-jobs)
11. [Tools & Libraries Reference](#11-tools--libraries-reference)
12. [Interview Questions on EDA](#12-interview-questions-on-eda)

---

## 1. What is EDA?

**Exploratory Data Analysis (EDA)** is the process of analyzing datasets to summarize their main characteristics — often with visual methods — *before* applying any machine learning model.

> **EDA = Understanding your data deeply before teaching a machine to learn from it.**

EDA was formalized by statistician **John Tukey** in 1977. It is the single most important step in any AIML project.

### EDA vs Other Steps

| Step | What It Does |
|------|-------------|
| **Data Collection** | Gather raw data |
| **EDA** | Understand, visualize, and clean data |
| **Feature Engineering** | Transform data for models |
| **Modeling** | Train ML/DL models |
| **Evaluation** | Measure model performance |

EDA sits at the heart of all these — it informs every downstream decision.

---

## 2. Why EDA Matters in AIML Jobs

In interviews and real-world ML roles, EDA skills signal:

- ✅ You don't blindly feed data into models
- ✅ You can catch data quality issues early
- ✅ You understand feature importance intuitively
- ✅ You can communicate insights to non-technical stakeholders
- ✅ You can design better features and architectures

> **"No amount of model complexity compensates for poor data understanding."**

---

## 3. The Titanic Dataset — Overview

We use the **Titanic Passenger Survival Dataset** from Kaggle throughout all levels.

### Loading the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# Load Titanic dataset
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame

# OR via seaborn
df = sns.load_dataset('titanic')

# OR from Kaggle CSV
df = pd.read_csv('train.csv')
```

### Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `survived` | int (0/1) | **Target**: Did the passenger survive? |
| `pclass` | int (1/2/3) | Ticket class (1=First, 2=Second, 3=Third) |
| `sex` | string | Gender of passenger |
| `age` | float | Age in years |
| `sibsp` | int | # of siblings/spouses aboard |
| `parch` | int | # of parents/children aboard |
| `fare` | float | Passenger fare (price paid) |
| `embarked` | string | Port of Embarkation (C/Q/S) |
| `class` | string | Ticket class (text version) |
| `who` | string | man/woman/child |
| `adult_male` | bool | Is adult male? |
| `deck` | string | Deck of cabin |
| `embark_town` | string | Town name of embarkation |
| `alive` | string | yes/no (same as survived) |
| `alone` | bool | Is passenger traveling alone? |

**Key facts:**
- 891 passengers (training set)
- 38.4% survival rate
- Mix of numerical, categorical, and boolean features
- Contains missing values (realistic!)

---

## 4. Level 1 — Beginner EDA

### 4.1 First Look at the Data

```python
# Shape: rows × columns
print(df.shape)          # (891, 15)

# First 5 rows
df.head()

# Last 5 rows
df.tail()

# Column names
df.columns.tolist()

# Data types
df.dtypes

# Quick summary
df.info()
```

**Output Interpretation:**
- `object` dtype = categorical/text
- `int64` / `float64` = numerical
- Non-null count tells you about missing values

---

### 4.2 Basic Statistics

```python
# Descriptive statistics for numerical columns
df.describe()
```

**What to look for in `.describe()`:**

| Statistic | What It Means |
|-----------|--------------|
| `count` | Non-missing values |
| `mean` | Average value |
| `std` | Spread of values |
| `min` / `max` | Range |
| `25%`, `50%`, `75%` | Quartiles (distribution shape) |

```python
# For Titanic:
# age: mean=29.7, std=14.5 → wide age range
# fare: mean=32.2, std=49.7 → highly skewed (some paid a LOT)
# survived: mean=0.384 → class imbalance (only 38% survived)
```

---

### 4.3 Checking Missing Values

```python
# Count of missing values per column
df.isnull().sum()

# Percentage of missing values
df.isnull().sum() / len(df) * 100

# Visual heatmap of missing values
import missingno as msno
msno.matrix(df)
plt.show()
```

**Titanic Missing Values:**
```
age        177  (19.9%)  ← Important! Must handle
deck       688  (77.2%)  ← Too much missing, likely drop
embarked     2  (0.2%)   ← Easy to fill
```

> **Rule of thumb:** >40% missing → consider dropping the column. <5% missing → safe to impute.

---

### 4.4 Value Counts for Categorical Columns

```python
# Survival count
df['survived'].value_counts()
# 0    549  (died)
# 1    342  (survived)

# Passenger class distribution
df['pclass'].value_counts()

# Sex distribution
df['sex'].value_counts()

# Embarkation port
df['embarked'].value_counts()
```

---

### 4.5 Your First Plot — Bar Chart

```python
import matplotlib.pyplot as plt

# Survival count plot
df['survived'].value_counts().plot(kind='bar', color=['salmon', 'steelblue'])
plt.title('Survival Count (0=Died, 1=Survived)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
```

---

### 4.6 Beginner Insights from Titanic

- 61.6% of passengers died
- Majority were in 3rd class (lower income)
- More males than females aboard
- Most passengers embarked from Southampton

---

## 5. Level 2 — Intermediate EDA

### 5.1 Univariate Analysis

**Univariate** = analyzing one variable at a time.

#### Numerical: Histogram + KDE

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Age distribution
df['age'].hist(bins=30, ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Age Distribution')

# Fare distribution
df['fare'].hist(bins=50, ax=axes[1], color='salmon', edgecolor='black')
axes[1].set_title('Fare Distribution')

# Survived distribution
df['survived'].hist(bins=3, ax=axes[2], color='green', edgecolor='black')
axes[2].set_title('Survived Distribution')

plt.tight_layout()
plt.show()

# KDE Plot (smoother version)
df['age'].dropna().plot(kind='kde', title='Age KDE')
plt.show()
```

#### Categorical: Count Plot

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.countplot(data=df, x='pclass', ax=axes[0])
sns.countplot(data=df, x='sex', ax=axes[1])
sns.countplot(data=df, x='embarked', ax=axes[2])

plt.tight_layout()
plt.show()
```

---

### 5.2 Bivariate Analysis

**Bivariate** = analyzing two variables together to find relationships.

#### Categorical vs Target (Survival Rate)

```python
# Survival rate by sex
df.groupby('sex')['survived'].mean()
# female    0.742
# male      0.189  ← Women had much higher survival!

# Survival rate by class
df.groupby('pclass')['survived'].mean()
# 1    0.630
# 2    0.473
# 3    0.242  ← 3rd class had lowest survival rate

# Survival rate by embarkation
df.groupby('embarked')['survived'].mean()
```

#### Grouped Bar Charts

```python
# Survival by sex and class
pd.crosstab(df['pclass'], df['sex'], values=df['survived'], aggfunc='mean').plot(
    kind='bar', figsize=(8, 5), color=['salmon', 'steelblue']
)
plt.title('Survival Rate by Class and Sex')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.legend(title='Sex')
plt.show()
```

#### Numerical vs Target: Box Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age vs Survival
sns.boxplot(data=df, x='survived', y='age', ax=axes[0])
axes[0].set_title('Age vs Survival')
axes[0].set_xticklabels(['Died', 'Survived'])

# Fare vs Survival
sns.boxplot(data=df, x='survived', y='fare', ax=axes[1])
axes[1].set_title('Fare vs Survival')
axes[1].set_xticklabels(['Died', 'Survived'])

plt.tight_layout()
plt.show()
```

#### Violin Plot (Box + Distribution)

```python
sns.violinplot(data=df, x='pclass', y='age', hue='survived', split=True)
plt.title('Age Distribution by Class and Survival')
plt.show()
```

---

### 5.3 Correlation Analysis

```python
# Correlation matrix (numerical columns only)
corr_matrix = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap — Titanic')
plt.show()
```

**Key Correlations in Titanic:**

| Pair | Correlation | Meaning |
|------|-------------|---------|
| `pclass` & `survived` | -0.34 | Higher class = higher survival |
| `fare` & `survived` | +0.26 | Higher fare = higher survival |
| `age` & `survived` | -0.08 | Weak (but children survived more) |
| `pclass` & `fare` | -0.55 | Higher class = more expensive |

> ⚠️ **Correlation ≠ Causation.** Higher fare passengers survived more because they were in better cabins (upper decks), NOT because fare itself saved them.

---

### 5.4 Outlier Detection

```python
# Box plots reveal outliers
df[['age', 'fare', 'sibsp', 'parch']].plot(kind='box', subplots=True, 
                                             figsize=(12, 4), layout=(1, 4))
plt.tight_layout()
plt.show()

# IQR Method
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['fare'] < lower_bound) | (df['fare'] > upper_bound)]
print(f"Fare outliers: {len(outliers)}")  # ~116 passengers paid extremely high fares

# Z-Score Method
from scipy import stats
z_scores = np.abs(stats.zscore(df['fare'].dropna()))
outliers_z = df[z_scores > 3]
```

**Titanic fare insight:** Max fare = £512 (a first-class suite). Most third-class tickets cost £7–£8.

---

### 5.5 Pivot Tables for Quick Insights

```python
# Survival rate by sex AND class
pivot = df.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='mean')
print(pivot)
#         pclass       1         2         3
# sex
# female          0.968     0.921     0.500
# male            0.369     0.157     0.135

# Visualization
sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn')
plt.title('Survival Rate by Sex and Class')
plt.show()
```

> **Key Insight:** First-class females had a 96.8% survival rate. Third-class males had only 13.5%.

---

## 6. Level 3 — Advanced EDA

### 6.1 Multivariate Analysis

```python
# Pair plot — all numerical features vs each other, colored by survival
sns.pairplot(df[['survived', 'age', 'fare', 'pclass', 'sibsp', 'parch']], 
             hue='survived', diag_kind='kde', palette='husl')
plt.suptitle('Pair Plot — Titanic', y=1.02)
plt.show()
```

#### FacetGrid — Multiple Plots by Category

```python
# Age distribution by survival, split by sex
g = sns.FacetGrid(df, col='sex', hue='survived', height=5)
g.map(sns.histplot, 'age', bins=20, alpha=0.6)
g.add_legend(title='Survived')
g.set_titles(col_template="{col_name}")
plt.show()
```

---

### 6.2 Feature Engineering Insights from EDA

EDA reveals **which new features to engineer**.

```python
# Family Size = sibsp + parch + 1 (self)
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Is alone?
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Survival rate by family size
df.groupby('family_size')['survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Family Size')
plt.show()
# Insight: Solo travelers and large families (>4) had lower survival
# Small families (2-4) had highest survival!

# Title extraction from Name
df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['title'].value_counts()
# Mr       517
# Miss     182
# Mrs      125
# Master    40
# Dr         7 ...

# Survival by title
df.groupby('title')['survived'].mean().sort_values(ascending=False)
# Mrs     0.793
# Miss    0.702
# Master  0.575
# Mr      0.157
```

---

### 6.3 Missing Value Patterns — Advanced

```python
import missingno as msno

# Dendrogram — correlates missingness patterns
msno.dendrogram(df)
plt.show()

# Heatmap of missing value co-occurrence
msno.heatmap(df)
plt.show()

# Insight: Are missing 'age' values random or systematic?
df[df['age'].isnull()]['pclass'].value_counts(normalize=True)
# 3rd class passengers more likely to have missing age!
# → This is NOT Missing At Random (MAR), it's informative missingness
```

**Types of Missingness:**

| Type | Definition | Titanic Example |
|------|-----------|-----------------|
| **MCAR** | Missing Completely At Random | Random data entry errors |
| **MAR** | Missing At Random (depends on other columns) | Lower class → less recorded age |
| **MNAR** | Missing Not At Random (depends on the value itself) | Very old/young passengers not recorded |

---

### 6.4 Distribution Analysis — Skewness & Kurtosis

```python
# Skewness and Kurtosis
for col in ['age', 'fare', 'sibsp', 'parch']:
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f"{col}: Skewness={skew:.2f}, Kurtosis={kurt:.2f}")

# fare: Skewness=4.79 (HIGHLY right-skewed), Kurtosis=33.4 (heavy tails)
```

**Skewness Guide:**

| Value | Interpretation |
|-------|---------------|
| 0 | Symmetric (Normal) |
| > 1 | High right skew (long right tail) |
| < -1 | High left skew (long left tail) |

```python
# Log transformation for skewed features
df['fare_log'] = np.log1p(df['fare'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['fare'].hist(bins=50, ax=axes[0], title='Fare (original, skewed)')
df['fare_log'].hist(bins=50, ax=axes[1], title='Fare (log-transformed, normalized)')
plt.tight_layout()
plt.show()
```

---

### 6.5 Interaction Effects

```python
# 3-way interaction: Sex × Class × Age → Survival
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, sex in enumerate(['male', 'female']):
    subset = df[df['sex'] == sex]
    for pclass in [1, 2, 3]:
        class_subset = subset[subset['pclass'] == pclass]
        axes[i].hist(class_subset[class_subset['survived']==1]['age'].dropna(),
                     bins=15, alpha=0.5, label=f'Class {pclass} - Survived')
    axes[i].set_title(f'{sex.capitalize()} Survivors by Class')
    axes[i].legend()
plt.tight_layout()
plt.show()
```

---

### 6.6 Statistical Tests in EDA

```python
from scipy import stats

# Chi-Square Test: Is sex related to survival?
contingency_table = pd.crosstab(df['sex'], df['survived'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi2={chi2:.2f}, p-value={p_value:.6f}")
# p < 0.05 → Yes, sex is statistically significantly related to survival

# T-Test: Is average age different between survivors and non-survivors?
survived_age = df[df['survived']==1]['age'].dropna()
died_age = df[df['survived']==0]['age'].dropna()
t_stat, p_val = stats.ttest_ind(survived_age, died_age)
print(f"T-statistic={t_stat:.2f}, p-value={p_val:.4f}")
# Survivors were slightly younger on average

# Mann-Whitney U (non-parametric alternative for non-normal distributions)
u_stat, p_val = stats.mannwhitneyu(
    df[df['survived']==1]['fare'].dropna(),
    df[df['survived']==0]['fare'].dropna()
)
print(f"Mann-Whitney U: p-value={p_val:.6f}")
# Survivors paid significantly higher fares
```

---

### 6.7 Advanced Visualization — Plotly (Interactive)

```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive scatter plot
fig = px.scatter(df.dropna(subset=['age', 'fare']),
                 x='age', y='fare', color='survived',
                 size='pclass', hover_data=['sex', 'pclass'],
                 title='Age vs Fare colored by Survival',
                 color_discrete_map={0: 'red', 1: 'green'})
fig.show()

# Sunburst chart — hierarchical breakdown
fig = px.sunburst(df.dropna(subset=['sex', 'pclass']),
                  path=['sex', 'pclass', 'survived'],
                  title='Survival Hierarchy: Sex → Class → Outcome')
fig.show()

# Parallel coordinates
fig = px.parallel_coordinates(
    df[['survived', 'pclass', 'age', 'fare', 'sibsp', 'parch']].dropna(),
    color='survived',
    color_continuous_scale=px.colors.diverging.Tealrose
)
fig.show()
```

---

## 7. Level 4 — EDA for ML/DL Pipelines

### 7.1 Target Variable Analysis (Classification)

```python
# Class imbalance check
class_counts = df['survived'].value_counts()
class_ratio = class_counts / len(df) * 100
print(class_ratio)
# 0    61.62%
# 1    38.38%

# Imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")  # ~1.6:1 (mild imbalance)

# For severe imbalance (>5:1), you'd use:
# - SMOTE (Synthetic Minority Over-sampling)
# - Class weights in model
# - Precision-Recall instead of Accuracy
```

---

### 7.2 Feature Importance via EDA (Before Modeling)

```python
# Point-Biserial Correlation (numerical vs binary target)
from scipy.stats import pointbiserialr

numerical_cols = ['age', 'fare', 'sibsp', 'parch', 'pclass']
for col in numerical_cols:
    corr, p_val = pointbiserialr(df[col].fillna(df[col].median()), df['survived'])
    print(f"{col}: r={corr:.3f}, p={p_val:.4f}")
```

```python
# Mutual Information (captures non-linear relationships too)
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Encode categoricals
df_encoded = df.copy()
le = LabelEncoder()
for col in ['sex', 'embarked', 'class', 'who']:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df_encoded[features].fillna(df_encoded[features].median())
y = df_encoded['survived']

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.Series(mi_scores, index=features).sort_values(ascending=False)

mi_df.plot(kind='bar', title='Mutual Information Scores vs Survival')
plt.ylabel('MI Score')
plt.show()
```

---

### 7.3 Preprocessing Needs Identified via EDA

```python
# After EDA, document your preprocessing plan:

preprocessing_plan = """
PREPROCESSING PLAN (informed by EDA):
=======================================

1. MISSING VALUES:
   - age (20% missing): Impute with median by pclass+sex group
   - embarked (0.2% missing): Impute with mode ('S')
   - deck (77% missing): Drop column

2. ENCODING:
   - sex: Binary encode (male=0, female=1)
   - embarked: One-hot encode (3 ports)
   - pclass: Keep as ordinal (already numeric)

3. SCALING:
   - fare: Log transform (skewness=4.79), then MinMaxScale
   - age: StandardScaler (normal-ish distribution)

4. FEATURE ENGINEERING (from EDA insights):
   - Create 'family_size' = sibsp + parch + 1
   - Create 'is_alone' = (family_size == 1)
   - Extract 'title' from 'name' column
   - Create 'fare_per_person' = fare / family_size

5. DROP COLUMNS:
   - name, ticket (too unique, no signal)
   - cabin/deck (too much missingness)
   - alive (data leakage — same as target)
"""
print(preprocessing_plan)
```

---

### 7.4 EDA for Deep Learning — Additional Checks

```python
# For DL models, also check:

# 1. Feature scale compatibility (important for gradients)
df[['age', 'fare', 'sibsp', 'parch']].describe()

# 2. Check for perfectly correlated features (redundancy)
corr = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].corr().abs()
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [(col, row) for col in upper_tri.columns 
             for row in upper_tri.index if upper_tri[col][row] > 0.8]
print("Highly correlated pairs:", high_corr)

# 3. Variance check — zero variance features are useless
zero_var_cols = [col for col in df.select_dtypes(include='number').columns 
                 if df[col].std() == 0]
print("Zero variance columns:", zero_var_cols)
```

---

### 7.5 Train/Validation Split Analysis

```python
from sklearn.model_selection import train_test_split

X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = df['survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# CRUCIAL: Verify class distribution is maintained after split
print("Train survival rate:", y_train.mean())
print("Val survival rate:", y_val.mean())
# Should both be ~0.384 (stratify=y ensures this)

# EDA on train set only (never peek at test data!)
print("Train age distribution:", X_train['age'].describe())
```

---

## 8. Level 5 — EDA for Generative AI

### 8.1 Text/NLP EDA on Titanic Names

The `name` column contains rich text data — useful for GenAI context.

```python
import re
from collections import Counter

# Basic text EDA
names = df['name'].dropna()

# Average name length
df['name_length'] = df['name'].str.len()
df.groupby('survived')['name_length'].mean()
# Survivors tended to have longer names (slightly more prestigious)

# Title extraction and frequency
df['title'] = df['name'].str.extract(r',\s*([A-Za-z]+)\.', expand=False)
title_counts = df['title'].value_counts()
print(title_counts)

# Rare titles → group as "Rare"
rare_titles = title_counts[title_counts < 10].index
df['title_grouped'] = df['title'].replace(rare_titles, 'Rare')

# Word frequency in names
all_words = ' '.join(names).lower().split()
word_freq = Counter(all_words)
# (Most common are common English surnames)
```

### 8.2 EDA for LLM Fine-Tuning Dataset Quality

When preparing the Titanic data for LLM fine-tuning (e.g., training a model to predict survival from passenger descriptions):

```python
# Create natural language descriptions from structured data
def passenger_to_text(row):
    """Convert a Titanic row into a natural language description."""
    gender = 'male' if row['sex'] == 'male' else 'female'
    class_map = {1: 'first', 2: 'second', 3: 'third'}
    pclass_text = class_map.get(row['pclass'], 'unknown')
    
    age_text = f"{int(row['age'])} years old" if pd.notna(row['age']) else "age unknown"
    
    family = []
    if row['sibsp'] > 0:
        family.append(f"{row['sibsp']} sibling(s)/spouse(s)")
    if row['parch'] > 0:
        family.append(f"{row['parch']} parent(s)/child(ren)")
    family_text = "traveling alone" if not family else "traveling with " + " and ".join(family)
    
    return (f"A {age_text} {gender} passenger traveling in {pclass_text} class, "
            f"who paid £{row['fare']:.1f} for their ticket and boarded at "
            f"{row.get('embark_town', 'an unknown port')}. They were {family_text}.")

# Apply to dataset
df['text_description'] = df.apply(passenger_to_text, axis=1)
print(df['text_description'].iloc[0])

# EDA on generated text
df['text_length'] = df['text_description'].str.split().str.len()
print(f"Average tokens per description: {df['text_length'].mean():.1f}")
print(f"Min: {df['text_length'].min()}, Max: {df['text_length'].max()}")

# Label for fine-tuning
df['label'] = df['survived'].map({0: 'did not survive', 1: 'survived'})
```

### 8.3 EDA for Prompt Engineering Data

```python
# Analyze what patterns lead to good model predictions
# → This informs how to structure prompts for GenAI

# Find most "decisive" passengers (very clear survival factors)
df['survival_score'] = (
    (df['sex'] == 'female').astype(int) * 0.4 +
    (df['pclass'] == 1).astype(int) * 0.3 +
    (df['age'] < 16).astype(int) * 0.2 +
    (df['fare'] > df['fare'].quantile(0.75)).astype(int) * 0.1
)

# High-certainty examples for few-shot prompting
good_examples = df[
    ((df['survival_score'] > 0.6) & (df['survived'] == 1)) |  # Clear survivors
    ((df['survival_score'] < 0.1) & (df['survived'] == 0))    # Clear non-survivors
]
print(f"High-certainty examples: {len(good_examples)}")
```

### 8.4 Embedding Space EDA

```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Prepare features for embedding analysis
features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
X_embed = df[features].fillna(df[features].median())
X_scaled = StandardScaler().fit_transform(X_embed)

# t-SNE for 2D visualization of feature space
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                      c=df['survived'], cmap='RdYlGn', alpha=0.6, s=30)
plt.colorbar(scatter, label='Survived')
plt.title('t-SNE of Titanic Feature Space — colored by Survival')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# UMAP (faster, better structure preservation)
# pip install umap-learn
import umap
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
# (similar plot)
```

> **GenAI insight:** If the embedding space shows clear cluster separation → a retrieval-augmented approach (RAG) can effectively find similar passengers to inform predictions.

---

## 9. Level 6 — EDA for Agentic AI Systems

### 9.1 EDA as an Agent Task

In **Agentic AI**, an LLM-based agent can autonomously perform EDA using tools. Here's how you'd design such a system:

```python
# Example: Agentic EDA using LangChain / tool calling

from anthropic import Anthropic

client = Anthropic()

# Define EDA tools that the agent can call
tools = [
    {
        "name": "get_basic_stats",
        "description": "Get basic statistics for a column",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Column name"},
                "stat": {"type": "string", "enum": ["mean", "median", "std", "missing_pct"]}
            },
            "required": ["column", "stat"]
        }
    },
    {
        "name": "get_correlation",
        "description": "Get correlation between two columns",
        "input_schema": {
            "type": "object",
            "properties": {
                "col1": {"type": "string"},
                "col2": {"type": "string"}
            },
            "required": ["col1", "col2"]
        }
    },
    {
        "name": "plot_distribution",
        "description": "Plot distribution of a column",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "split_by": {"type": "string", "description": "Grouping column"}
            },
            "required": ["column"]
        }
    }
]

# Tool execution
def execute_tool(tool_name, tool_input, df):
    if tool_name == "get_basic_stats":
        col = tool_input["column"]
        stat = tool_input["stat"]
        if stat == "mean":
            return str(df[col].mean())
        elif stat == "missing_pct":
            return str(df[col].isnull().mean() * 100)
    elif tool_name == "get_correlation":
        return str(df[tool_input['col1']].corr(df[tool_input['col2']]))
    return "Result computed"

# Agent loop
def run_eda_agent(df, user_question):
    messages = [{"role": "user", "content": user_question}]
    
    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input, df)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        # Continue conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

---

### 9.2 Data Quality Scoring (Agentic Decision Making)

Agents need to programmatically assess data quality before proceeding:

```python
def compute_data_quality_score(df):
    """
    Compute an overall data quality score (0-100).
    An agent uses this to decide next steps.
    """
    scores = {}
    
    # 1. Completeness (missing values)
    missing_pct = df.isnull().mean().mean()
    scores['completeness'] = (1 - missing_pct) * 100
    
    # 2. Uniqueness (duplicate rows)
    dup_pct = df.duplicated().sum() / len(df)
    scores['uniqueness'] = (1 - dup_pct) * 100
    
    # 3. Consistency (mixed types in columns)
    type_issues = 0
    for col in df.select_dtypes(include='object').columns:
        try:
            pd.to_numeric(df[col])
            type_issues += 1  # String column that could be numeric
        except:
            pass
    scores['consistency'] = max(0, 100 - (type_issues / len(df.columns)) * 100)
    
    # 4. Validity (values in expected ranges)
    validity_issues = 0
    if 'age' in df.columns:
        validity_issues += (df['age'] < 0).sum() + (df['age'] > 120).sum()
    if 'fare' in df.columns:
        validity_issues += (df['fare'] < 0).sum()
    scores['validity'] = max(0, 100 - (validity_issues / len(df)) * 100)
    
    overall = np.mean(list(scores.values()))
    
    return {
        'overall_score': overall,
        'dimension_scores': scores,
        'recommendation': 'proceed' if overall > 70 else 'clean_first'
    }

quality = compute_data_quality_score(df)
print(quality)

# Titanic output:
# overall_score: ~82
# recommendation: 'proceed' (with imputation for age)
```

---

### 9.3 Automated EDA Report Generation

```python
import json
from datetime import datetime

def generate_eda_report(df, target_col='survived'):
    """Generate structured EDA report for agent consumption."""
    
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "dataset": "Titanic",
            "shape": list(df.shape),
            "target_column": target_col
        },
        "data_quality": {
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_rows": len(df)
        },
        "target_analysis": {
            "class_distribution": df[target_col].value_counts().to_dict(),
            "class_balance_ratio": float(df[target_col].value_counts().iloc[0] / 
                                         df[target_col].value_counts().iloc[1])
        },
        "numerical_features": {},
        "categorical_features": {},
        "correlations": {}
    }
    
    # Numerical features
    for col in df.select_dtypes(include='number').columns:
        if col != target_col:
            report["numerical_features"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "skewness": float(df[col].skew()),
                "missing_pct": float(df[col].isnull().mean() * 100),
                "correlation_with_target": float(df[col].corr(df[target_col]))
            }
    
    # Categorical features
    for col in df.select_dtypes(include='object').columns:
        report["categorical_features"][col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(5).to_dict(),
            "missing_pct": float(df[col].isnull().mean() * 100)
        }
    
    return json.dumps(report, indent=2)

report_json = generate_eda_report(df)

# This JSON report can be fed to an LLM agent for insights:
# "Based on this EDA report, what features should I engineer? 
#  What preprocessing steps do I need? What models would work best?"
```

### 9.4 EDA Feedback Loop in Multi-Agent Systems

```
┌─────────────────────────────────────────────────────┐
│              MULTI-AGENT EDA PIPELINE               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Data Ingestor Agent]                              │
│       │ loads Titanic CSV                           │
│       ▼                                             │
│  [EDA Agent]                                        │
│       │ runs statistical tests                      │
│       │ generates visualizations                    │
│       │ scores data quality                         │
│       ▼                                             │
│  [Insight Agent]                                    │
│       │ interprets EDA results                      │
│       │ generates natural language insights         │
│       ▼                                             │
│  [Feature Engineering Agent]                        │
│       │ creates new features based on EDA insights  │
│       ▼                                             │
│  [Validation Agent]                                 │
│       │ checks engineered features against EDA      │
│       │ flags data leakage                          │
│       ▼                                             │
│  [Report Agent]                                     │
│       │ compiles full EDA report                    │
│       │ → sends to human or next ML pipeline        │
└─────────────────────────────────────────────────────┘
```

```python
# Example: Insight Agent using Claude API
import anthropic

def get_ai_insights_from_eda(eda_report_json):
    """Use Claude to generate insights from structured EDA report."""
    client = anthropic.Anthropic()
    
    prompt = f"""You are an expert data scientist. 
    Analyze this EDA report and provide:
    1. Top 3 most important features for predicting survival
    2. Key data quality concerns
    3. Recommended preprocessing steps
    4. Suggested feature engineering ideas
    5. Potential model architectures to try
    
    EDA Report:
    {eda_report_json}
    
    Be specific and reference actual values from the report."""
    
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

insights = get_ai_insights_from_eda(report_json)
print(insights)
```

---

## 10. EDA Checklist for AIML Jobs

Use this as your step-by-step guide for any new dataset:

### ✅ Phase 1: First Look
- [ ] Load data and check shape
- [ ] View first/last few rows
- [ ] Check data types of all columns
- [ ] Print `.info()` and `.describe()`
- [ ] Identify target variable and its type (classification/regression)

### ✅ Phase 2: Data Quality
- [ ] Count and visualize missing values per column
- [ ] Check for duplicate rows
- [ ] Identify impossible values (negative age, etc.)
- [ ] Classify missingness type (MCAR/MAR/MNAR)
- [ ] Plan imputation strategy for each column

### ✅ Phase 3: Univariate Analysis
- [ ] Histogram/KDE for each numerical feature
- [ ] Count plots for each categorical feature
- [ ] Compute skewness and kurtosis
- [ ] Identify outliers using IQR and Z-score
- [ ] Check target variable distribution (class balance)

### ✅ Phase 4: Bivariate Analysis
- [ ] Numerical vs Target: Box plots, violin plots
- [ ] Categorical vs Target: Group bar charts, pivot tables
- [ ] Numerical vs Numerical: Scatter plots
- [ ] Correlation heatmap for all numerical features
- [ ] Statistical significance tests (Chi-square, t-test)

### ✅ Phase 5: Multivariate Analysis
- [ ] Pair plots
- [ ] FacetGrid plots
- [ ] 3-way interactions
- [ ] t-SNE/UMAP for high-dimensional data

### ✅ Phase 6: Insights & Decisions
- [ ] Document top 5 EDA findings
- [ ] List features to engineer (with rationale)
- [ ] Define preprocessing plan
- [ ] Identify potential data leakage
- [ ] Plan model selection based on findings

---

## 11. Tools & Libraries Reference

### Core EDA Libraries

| Library | Use Case | Key Functions |
|---------|----------|---------------|
| **pandas** | Data manipulation | `describe()`, `value_counts()`, `groupby()`, `pivot_table()` |
| **numpy** | Numerical ops | `mean()`, `std()`, `percentile()` |
| **matplotlib** | Static plotting | `hist()`, `plot()`, `scatter()` |
| **seaborn** | Statistical plots | `heatmap()`, `pairplot()`, `violinplot()`, `countplot()` |
| **plotly** | Interactive plots | `express.scatter()`, `express.sunburst()` |
| **missingno** | Missing value viz | `matrix()`, `heatmap()`, `dendrogram()` |
| **scipy** | Statistical tests | `chi2_contingency()`, `ttest_ind()`, `stats.zscore()` |
| **sklearn** | ML preprocessing | `mutual_info_classif()`, `train_test_split()` |

### Advanced EDA Libraries

| Library | Use Case |
|---------|----------|
| **ydata-profiling** (formerly pandas-profiling) | Auto-generate full EDA report in one line |
| **sweetviz** | Compare train vs test distributions |
| **dtale** | Interactive EDA web interface |
| **umap-learn** | Dimensionality reduction for embedding visualization |
| **shap** | Feature importance visualization |
| **eli5** | Explain model predictions |

### One-Line EDA Reports

```python
# ydata-profiling (auto EDA)
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Titanic EDA Report", explorative=True)
profile.to_file("titanic_eda.html")

# sweetviz (train vs test comparison)
import sweetviz as sv
report = sv.analyze(df)
report.show_html("titanic_sweetviz.html")
```

---

## 12. Interview Questions on EDA

### Beginner Level

**Q: What is EDA and why is it important?**
> EDA is the process of analyzing data to understand its structure, patterns, and quality before modeling. It's important because garbage in = garbage out. Without EDA, you risk training models on dirty, imbalanced, or poorly understood data.

**Q: How would you handle missing values in the age column?**
> First, check how much is missing (19.9%). Then understand why: in Titanic, 3rd class passengers had more missing ages, suggesting MAR. Good strategy: impute with median age grouped by pclass and sex — this preserves demographic patterns.

**Q: What does the correlation coefficient tell you?**
> It measures linear relationship strength between two variables, ranging from -1 to +1. +1 = perfect positive relationship, -1 = perfect negative, 0 = no linear relationship. Important: it only captures linear relationships.

---

### Intermediate Level

**Q: What is the difference between correlation and causation? Give an example.**
> Correlation means two variables move together. Causation means one causes the other. In Titanic, fare and survival are correlated (r=0.26), but paying more doesn't cause you to survive — it's because first-class cabins were on upper decks with easier lifeboat access.

**Q: How do you detect and handle outliers?**
> Detection: IQR method (flag values below Q1-1.5×IQR or above Q3+1.5×IQR), Z-score method (|z|>3). Handling options: (1) Remove if data entry errors, (2) Cap/winsorize, (3) Log transform to reduce impact, (4) Use robust models like tree-based methods that handle outliers naturally.

**Q: What is class imbalance and how does EDA reveal it?**
> Class imbalance is when one target class has far more samples than another. EDA reveals it via `value_counts()` and `value_counts(normalize=True)`. For Titanic: 61.6% vs 38.4% (mild imbalance). Severe imbalance (>10:1) requires SMOTE, class weights, or F1-score metrics.

---

### Advanced Level

**Q: Explain the types of missing data and how each should be handled.**
> MCAR (Missing Completely At Random): Safe to drop rows or impute with mean/median. MAR (Missing At Random): Impute using related columns (e.g., age imputed using pclass+sex). MNAR (Missing Not At Random): The missing value itself carries information — consider adding a binary "was_missing" flag feature.

**Q: How would you perform EDA to detect data leakage?**
> Data leakage occurs when test-time information is used during training. EDA steps: (1) Check correlation of all features with target — suspiciously high correlation (>0.9) may indicate leakage, (2) Examine time-based features — ensure no future data is included, (3) Check that target-derived features aren't in the dataset (e.g., "alive" in Titanic = same as "survived"), (4) Verify train/validation distributions match.

**Q: How does EDA differ for GenAI vs classical ML projects?**
> For classical ML: focus on feature correlation, missing values, class balance, outliers. For GenAI: additionally analyze text quality (token length distribution, language variety), embedding space structure (t-SNE/UMAP), label noise in instruction-tuning datasets, and train/eval distribution shift. GenAI EDA also involves checking for PII, bias patterns in labels, and diversity of examples.

**Q: How would you use EDA insights to design an agentic AI pipeline?**
> EDA informs agent design in several ways: (1) Data quality score determines if a cleaning agent is needed before modeling, (2) Missing value patterns determine which imputation tools to give the agent, (3) Correlation structure informs the feature engineering agent's search space, (4) Class imbalance insights determine the evaluation metric the agent should optimize for, (5) Distribution analysis helps the validation agent flag out-of-distribution inputs at inference time.

---

## Summary: EDA Mental Model

```
Raw Data
   │
   ▼
[UNDERSTAND] → Shape, types, basic stats
   │
   ▼
[CLEAN] → Missing values, duplicates, outliers
   │
   ▼
[EXPLORE] → Univariate → Bivariate → Multivariate
   │
   ▼
[HYPOTHESIZE] → What patterns exist? Why?
   │
   ▼
[VALIDATE] → Statistical tests to confirm/reject
   │
   ▼
[ENGINEER] → What new features does EDA suggest?
   │
   ▼
[DOCUMENT] → Share insights, preprocessing plan
   │
   ▼
[MODEL READY ✅]
```

---

*Notes prepared using the **Titanic Passenger Survival Dataset** (Kaggle/Seaborn).  
All code examples are self-contained and executable with standard AIML libraries.*

*Dataset: 891 passengers, 15 features, binary classification target (`survived`)*

---

> **Pro Tip for AIML Interviews:** Don't just show that you can run code. Show that you can *interpret* what you see, *question* why it exists, and *decide* what to do about it. That's the difference between a data analyst and an ML engineer.
