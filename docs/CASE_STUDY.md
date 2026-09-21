# Case Study — DeepLearning-Charity

**Repository:** [DeepLearning-Charity](https://github.com/Freddricklogan/DeepLearning-Charity) · **Live report:** [freddricklogan.github.io/DeepLearning-Charity](https://freddricklogan.github.io/DeepLearning-Charity/) · **Author:** Freddrick Logan

---

## 1. Who has this problem

Anyone who has to judge a machine-learning result from a repository rather than a conversation: hiring managers reviewing a data-science portfolio, instructors grading a tabular-classification assignment, and the analysts at a foundation who might be asked whether a model like this should influence a funding decision. The question they all ask is the same: compared with what, measured how, and would it repeat.

## 2. The problem, as a scenario

A reviewer opens the repository. The README promises hyperparameter optimisation for maximum performance and production-ready deployment. She looks for a number and finds none. The code is a single 600-line file that fits its scaler on the whole dataset, loops over a grid of architectures for fifty epochs each and prints results to a terminal that no longer exists, and expects a CSV that is not in the repository and is not referenced anywhere. A notebook has no saved outputs. She cannot tell whether the network beat a coin flip, let alone logistic regression. That was the earlier version of this repository.

## 3. What it costs to leave it alone

For the author, a machine-learning project with no measured result is a liability in a portfolio: it invites the one question it cannot answer. For anyone who might reuse the pipeline, a scaler fitted before the split leaks test information into training and makes every reported number optimistic. For the domain, a model presented as a predictor of which charities use money well, without a baseline, a calibration check or a statement of limits, is the kind of artefact that gets misused.

## 4. The approach, and the alternative I rejected

I rejected keeping the notebook and adding a results cell. A saved output is a claim, not evidence, and the leakage would have remained. The pipeline became a package of small, typed, tested functions. `data.py` loads the vendored dataset and validates its schema, folds rare categories under explicit thresholds and records what it folded, log-transforms the ask amount, one-hot encodes with named features, splits with stratification and fits the scaler on training rows only. `baseline.py` fits logistic regression on the same features. `network.py` builds one stated Keras 3 architecture on the JAX backend and trains it with early stopping, restoring the best weights. `evaluate.py` computes accuracy, precision, recall, F1, ROC AUC, confusion counts, a reliability table, a threshold sweep and ROC points from labels and probabilities, with tests against scikit-learn. `report.py` renders a static page and a model card from those results, and CI runs the whole thing and publishes the output, so the numbers on the page are the numbers of the run.

## 5. What the code does today

`charity-model report --out dist --seed 42` trains both models and writes three files: a report page with the Executive Shell, a JSON file of every metric, and a model card. The page states the data source and the preprocessing that was applied, puts the network's metrics beside the baseline's and the majority-class rate, draws the training and validation loss by epoch and both ROC curves, and shows the reliability table and the threshold sweep. The model card records the architecture, training procedure, data handling, evaluation table and limits, and ends with the command that regenerates it. A Streamlit companion trains once and lets a reader move the decision threshold to see precision, recall and the confusion counts change on the held-out rows. A Dockerfile builds the report in a container.

## 6. Evidence

Thirteen tests at 100 % statement coverage cover schema validation and rejection, rare-category folding at and above the threshold, the classic result of nine application types after binning, the log-transformed range, the no-leakage property of the scaler, stratification, the metrics against scikit-learn on random data and on a single-class edge case, the reliability bins and threshold sweep monotonicity, the ROC endpoints, the baseline beating the majority rate, deterministic network training with the expected parameter count, and the report writer's outputs. On seed 42 with 8,575 held-out rows and a 53.24 % positive rate, logistic regression scored accuracy 0.7186 and AUC 0.7587; the network scored 0.7236 and 0.7862 after 30 epochs with the best at epoch 29. The report rendered with zero console errors and no horizontal scroll at 1280 or 400 pixels. `AUDIT.md` records eleven findings against the earlier build.

## 7. What it would take to run this in production

It should not run in production as a decision system, and the card says so: the features describe applications, not outcomes, and the label's provenance is the exercise's. As a pipeline pattern, the additions would be repeated runs across seeds with confidence intervals, a held-out set separated by time rather than at random, feature-importance and error analysis by category, and a review of what "successful" meant to whoever labelled it. The package would take a new dataset through the same loader and validator.

## 8. Limits and next steps

One split, one seed, one architecture; the half-point accuracy gap over logistic regression is not evidence of anything until it repeats. No fairness analysis is possible on these features. Next, in order: a multi-seed run reported with intervals, a permutation-importance table, and calibration correction if the reliability table warrants it.

## 9. Who should look at this

**Hiring manager:** evidence that I report machine-learning results against a baseline, measured in the pipeline that publishes them, with the limits written down.
**Consulting client:** a template for how a model should be presented before anyone acts on it.
**Engineer:** read `src/charity_model/data.py` with `tests/test_data.py` for the leakage-free preprocessing, and `report.py` for how the card is generated.
