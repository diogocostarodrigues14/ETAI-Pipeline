20260672 - Diogo Miguel Costa Rodrigues

Week 1- The logistic regression is better than this decision tree because we didn´t put max_depht on our tree which led us to overfitting on that case. I saw this by comparing the train with the test accuracy.

Week 2- This week, I implemented ,in the pipeline, the data cleaning function and put a new paramenter of max_depth=5 in the decision tree model.
By putting max_depth in the decision tree, i think i can say that i solved the overfitting problem because the difference between is now minimal. 
      Last week difference: +0.199
      This week difference: +0.012 (without cleaning) and +0.049 (with cleaning)


Initially, the uncleaned test set was with 1252 observations and had some data intigrity issues, like formatting duplicates and invalid or missing categories. The cleaning process was essential to fix these anomalies because it standardized the class labels and removed inconsistent records, which reduced the test set to 1188 observations.

After cleaning, by evaluating both algorithms, we can conclude that the decision_tree is  the best model for this case.
The tree achieved a higher test accuracy than the the logistic regression (0.67 vs 0.65), more stable generalization and a superior recall of 0.65 (vs 0.51)


Week 3- This week, I implemented in the pipeline the drop_duplicate_rows, build_preprocessor and build_model functions. I also implemented two new models : the dummy classifier and random forest.
In preprocessing this week I implemented the imputation with the median for numeric features and the most frequent for categorical features.
Comparing the 4 models, we can conclude that the worst one was the dummy classifier, as expected. The other tree got almost the same accuracy (0.66 vs 0.66 vs 0.67) but the random forest got the best f1-score, so i think i can conclude that i was the best model with this week´s implementations.
The four models weren´t overfitting as well.

Week 5-
### Cross Valitation
To ensure sound evaluation, the dataset is initially split into a development set and a locked test set that remains completely untouched during all experimentation and hyperparameter tuning. Within the development set, we evaluate models using stratified 5-fold cross-validation, which preserves the original class distribution across every split and provides a reliable measure of performance stability across folds.

| Model | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|
| Dummy | 0.550 | 0.549 ± 0 | 0 |
| Logistic regression | 0.658 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.665 | 0.674 ± 0.018 | 0.011 |
| Random forest | 0.675 | 0.681 ± 0.012| +0.008 |

I trust the 5-fold cross-validation accuracy much more than the old holdout score because a single split can easily be lucky or unlucky. Looking at the train-val gaps, all models show very small train–val gap, which means none of them are really overfitting.
The ranking from the previous week still holds under cross-validation: Random Forest is still the best model at around 0.681, followed closely by Decision Tree and Logistic Regression. Even so, the CV results show that Decision Tree and Logistic Regression are basically tied within the standard deviation, so having cross-validation gives a much safer comparison while keeping the test set locked. I also used max_depht=5 in the tree and in the forest and n_estimators=100 in the forest.


### Hyperparameter tuning

We used Optuna with 30 trials over 5 inner folds to optimize model-specific search spaces inside an outer 5-fold cross-validation scheme. This nested design prevents data leakage because all preprocessing, feature transformations, and hyperparameter selections are fitted exclusively on the inner training folds, leaving the outer validation folds completely untouched.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.015 | 0.679 | +0.002 | max_depth: 6, min_samples_leaf: 101, criterion: gini |
| Logistic regression | 0.672 ± 0.013 | 0.671 ± 0.015 | 0.677 | +0.004 | C: 0.075 |
| Random forest | 0.647 ± 0.016 | 0.682 ± 0.015 | 0.683 | -0.001 | max_depth: 10, min_samples_leaf: 43, n_estimators: 64, criterion: gini |

Tuning brought clear performance gains to both the Decision Tree (rising from 0.610 to 0.672) and Random Forest (improving from 0.647 to 0.682), with both gains easily exceeding their fold-to-fold standard deviations of approximately 0.015 to 0.018. In contrast, Logistic Regression remained with almost thge same accuracy (0.672 vs 0.671), indicating that standard L2 regularization was already close to optimal for this linear feature space. 
This tuning process changed the top-performing model: while Logistic Regression was the strongest baseline under default settings, Random Forest took the lead once its tree depth and leaf constraints were properly optimized.
For our final model (Random Forest), we report the honest nested CV score of 0.682 ± 0.015 rather than the optimistic best-trial score of 0.683, ensuring our performance estimate is entirely free from hyperparameter selection bias.


### Ensembles

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.015 (+0.008) | 0.679 | max_depth: 6, min_samples_leaf: 101, criterion: gini |
| Random forest | 0.647 ± 0.016 (+0.085) | 0.682 ± 0.015 (+0.006) | 0.683 | max_depth: 10, min_samples_leaf: 43, n_estimators: 64, criterion: gini |
| Oblique random forest | 0.645 ± 0.013 (+0.114) | 0.682 ± 0.014 (+0.008) | 0.682 | max_depth: 5, min_samples_leaf: 3, n_components: 9 |
| AdaBoost | 0.668 ± 0.013 (+0.003) | 0.682 ± 0.017 (+0.004) | 0.684 | max_depth: 4, n_estimators: 220, learning_rate: 0.025 |

I choose the standard Random Forest: while all three tuned ensembles reached an identical honest nested CV accuracy of 0.682, Random Forest provides higher stability with a lower fold-to-fold standard deviation (±0.015 vs ±0.017 for AdaBoost) and significantly lower computational overhead. 
Its lead over the other ensemble models is effectively zero and well within the ~0.015 fold-to-fold standard deviation, showing that the three architectures reached an identical performance plateau. The tuned hyperparameters align closely with theory: bagging architectures (Random Forest and Oblique Random Forest) utilize deeper trees (depths 5–10) because averaging across independent trees cancels out individual variance, whereas AdaBoost relies on shallower base trees (depth 4) paired with a conservative learning rate (0.025) to sequentially reduce bias without overfitting. Consequently, I report Random Forest with its honest nested cross-validation score of 0.682 ± 0.015 (rather than the optimistic 0.683 from the best tuning trial) to guarantee an unbiased estimate of generalization performance, as I recommended in the hyperparameter tuning section.



# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
