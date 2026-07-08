# Survey Item Importance

An end-to-end classification pipeline on Likert-scale survey responses, built to answer a question every survey team eventually asks: **which items in this questionnaire actually carry signal, and which ones are dead weight?**

The demo uses a regularized logistic regression trained on a public psychometric dataset. It applies stratified cross-validation, bootstrap confidence intervals, permutation-based item importance, and two independent diagnostic checks (model swap, label shuffling). The output is a ranked list of items by predictive contribution, which is the same output a consumer insights team needs when deciding whether a survey can be shortened without losing information.

## Why this matters for survey and consumer research

Practical use cases where the same pipeline transfers directly:

- **Survey shortening.** Rank items by permutation importance, drop the low-signal tail, re-estimate. Trade respondent fatigue for information density with a defensible criterion instead of a hunch.
- **Item validation for new instruments.** Before locking a market research questionnaire into fieldwork, run it against a labeled pilot sample and check whether the items are doing work individually or just co-varying.
- **Segment prediction from behavior.** Any categorical target that a business cares about (NPS bucket, purchase intent tier, product-fit segment, churn class) fits the same pipeline shape without changes to the code.
- **Model auditing.** The two diagnostic steps at the end (swap the classifier, shuffle the labels) are cheap sanity checks that catch overfitting and label leakage. Both are common failure modes in survey ML that aren't always caught by a train/test split alone.

The target in this demo is a demographic variable (see "Responsible use" below), used as a stand-in for a generic categorical target because the dataset is public and labeled. The methodological choices are what generalize, not the specific prediction task.

## Data

The [Open Sex Role Inventory (OSRI)](https://openpsychometrics.org/tests/OSRI/development/) collected 44 Likert-scale items (1-5 scale) from voluntary online respondents between 2015 and 2019, alongside self-reported demographics. Items cover everyday behaviors and preferences ("I use lotion on my hands", "I have thrown knives, axes or other sharp things", "I bake sweets just for myself sometimes"). The full codebook is in `data/codebook.txt`.

The CSV isn't shipped in this repo. You can download it from openpsychometrics.org and drop it into `data/` if you want to re-run the notebook.

## Pipeline

The full run is in `notebook/pipeline.ipynb`. In order:

1. **Load and filter.** Keep only respondents with a labeled binary target. Downsample to 25% for tractability during iteration (the pipeline scales without modification).
2. **Feature selection.** Twenty-six items were retained from the original 44 based on prior inspection. Feature correlation is checked to flag redundancy.
3. **Split and scale.** 75/25 train/test split, `StandardScaler` fit on the training set only.
4. **Hyperparameter tuning.** 5-fold stratified cross-validation over the L2 regularization strength `C`, sweeping ten orders of magnitude from `1e-5` to `1e4`. Scored with Matthews Correlation Coefficient (see "Technical notes" for why MCC over accuracy).
5. **Fit.** Best `C` chosen from the CV curve, model refit on the full training set.
6. **Bootstrap CI.** 100 resamples of 25% of the test set, MCC recomputed per resample. Reported as median and 5-95 percentile band.
7. **Permutation importance.** For each feature, values are randomly permuted and the drop in test MCC is recorded across 30 repeats. Features that don't matter for the prediction contribute noise around zero; features that do matter produce a visible, positive drop.
8. **Diagnostic checks.** The model is swapped to a linear SVC (same regularization, different loss) to confirm the result isn't specific to logistic regression. Then the test labels are shuffled and the model is re-evaluated on the shuffled labels; MCC should collapse to zero if the earlier performance was genuine and not the result of a leak.

## Results

### Hyperparameter tuning

![Cross-validation MCC across five folds, plotted against the L2 regularization strength C.](figures/cv_hyperparameter_tuning.png)

MCC on validation folds as a function of `C`. The curves are almost flat for `C` above `0.01`, which is the signature of a well-regularized fit on Likert data: the coefficient magnitudes are being clamped by the penalty, so additional flexibility doesn't buy anything. `C = 0.001` sits at the shoulder of the curve and was chosen to keep the model conservative. All five folds land within a narrow band (0.55 to 0.62 MCC), indicating stable performance across resamplings of the training set.

### Permutation importance across all items

![Boxplot of permutation-based item importance for all 26 features, computed across 30 shuffles on the test set.](figures/feature_importance_boxplot.png)

Distribution of permutation importance for every retained item, one boxplot per feature, computed by shuffling that feature 30 times on the test set and averaging the MCC drop. Most items sit near zero: shuffling them doesn't change the prediction, which means the model isn't using them. A handful stand out with visible, consistently positive contributions, and one item (position 2) dominates. This is the kind of chart that tells a survey designer, at a glance, which questions are earning their spot.

### Top items ranked

![Top ten items ranked by mean permutation importance on the test set.](figures/top_features_barplot.png)

The top ten items, sorted. When mapped back to their questionnaire content, the ranking is highly interpretable: items about clothing and cosmetics, weapons and physical risk, handmade gifts, and taking apart machines are all doing most of the work. That interpretability is the second thing a survey team wants, alongside the ranking itself: the ability to look at a top-item list and understand *why* the signal lives there.

### Headline performance

Test-set MCC of roughly 0.60 with a tight bootstrap CI. The label-shuffle diagnostic collapses this to approximately zero, as expected, which confirms the signal is real. The SVC diagnostic returns a comparable MCC, which confirms the finding isn't an artifact of the specific classifier.

## Technical notes

A few choices worth calling out for a technical reviewer:

- **MCC over accuracy.** Matthews Correlation Coefficient is bounded in `[-1, 1]`, symmetric under class relabeling, and doesn't get fooled by class imbalance. Accuracy would look artificially high on any survey with a majority segment. MCC also gives a natural sanity check: a well-behaved random baseline sits at zero, not at "1 minus majority-class proportion", which makes the label-shuffle diagnostic legible.
- **Permutation importance over model coefficients.** For a linear model with correlated Likert items, raw coefficient magnitudes can be misleading (they redistribute across correlated features under regularization). Permutation importance measures actual predictive contribution on held-out data and is model-agnostic.
- **Bootstrap CI on the test set.** A point estimate on a single train/test split can be optimistic or pessimistic by chance. Resampling the test set 100 times and reporting a CI gives a defensible range instead of a single number.
- **Two diagnostic steps rather than one.** Swapping the classifier catches model-specific artifacts. Shuffling labels catches leakage and data pipeline bugs. Neither alone would catch both failure modes.
- **Regularization sweep on log scale.** Ten orders of magnitude across `C` is more than the eventual answer needs, but it makes the shape of the curve visible and the choice defensible. Sweeping only a narrow band around a guessed optimum hides whether the model is actually plateauing or still climbing.

## Responsible use

The target variable in this demo is self-reported binary gender, drawn from a public psychometric dataset where respondents opted in and provided the labels themselves. It's used here as a stand-in for a generic categorical target because it's clean, labeled, and publicly available.

Predicting demographic attributes from behavioral data in an applied setting is a different question, one that carries fairness, consent, and regulatory implications (GDPR Art. 9, EU AI Act, sector-specific rules) and shouldn't be deployed without a specific lawful basis and a documented impact assessment. The value of this notebook is the shape of the pipeline, not the deployment of this particular classifier.

## Repo structure

```
survey-item-importance/
├── notebook/
│   └── pipeline.ipynb          Full run, from load to diagnostic checks
├── data/
│   └── codebook.txt            Item wording and demographic field encoding
├── figures/
│   ├── cv_hyperparameter_tuning.png
│   ├── feature_importance_boxplot.png
│   └── top_features_barplot.png
└── README.md
```

The path variable in the notebook is a placeholder (`path/to/data.csv`) and needs pointing at a local copy of the OSRI CSV. If you'd like a walkthrough or a runnable version with the data configured, get in touch.
