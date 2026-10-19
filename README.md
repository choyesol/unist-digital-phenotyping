# Social app use and depressive and anxiety symptoms: analysis code

This repository contains the analysis code for a 6-week observational study of social app use and depressive and anxiety symptoms among college students.

The scripts do two things:

1. They turn the Android app-use logs and the questionnaire data into analysis data sets.
2. They compute every statistic reported in the article, including the values shown in Figures 1 and 2.

## Data availability

The data are not in this repository and are not publicly available because of participant privacy. They are available from the corresponding author of the article on reasonable request.

The app category codebook is not included. Multimedia Appendix 2 of the article gives the definitions and examples of the categories.

## Requirements

- Python 3.11.9.
- The package versions in `requirements.txt`. Install them with:

  ```
  pip install -r requirements.txt
  ```

## Input files

Put the input files in one folder, for example `data/`. File names are given on the command line.

| Input | Format | Columns used |
|---|---|---|
| Android app-use logs, one file per recruitment wave | CSV, UTF-8 | `userId`; `appLogRecordDate`; `appName`; `appRunTimeForeground` (foreground seconds per app) |
| Daily study master file | Excel, sheet `Data` | `userId`; `deviceType` (`Android` or `iOS`); `Date` (one row per participant and study day) |
| Questionnaire file | Excel, first sheet | `ID`; `연구` (recruitment wave: `1차` or `2차`); `Age` (date of birth); `Sex` (`여` = female); `Week` (0, 2, 4 or 6); `Date` (completion date as mmdd in 2023 or yymmdd); `PHQ`; `GAD`; `SIAS`; `PSO`; `Smartphone` (K-SAPS); `GSDS` |

The codebook is a CSV file (UTF-8) with two columns, `appName` and `appCategory`. `appCategory` is `Community` (anonymous community), `SNS` (acquaintance-based social networking service) or `Messenger`. Apps not in the codebook are not assigned to a category but count toward the total logged app time. Apps are matched by name. Matching ignores letter case, repeated spaces and invisible formatting characters. Everytime minutes are taken from the apps named `Everytime` or `에브리타임`. Example rows:

| appName | appCategory |
|---|---|
| Everytime | Community |
| Instagram | SNS |
| KakaoTalk | Messenger |

## Run order

```
python 01_prepare.py --data-dir data --out-dir output --app-logs app_log_wave1.csv app_log_wave2.csv --master daily_master.xlsx --survey questionnaires.xlsx --codebook app_categories.csv
python 02_models.py --out-dir output
```

On a laptop, the two scripts take about 1 minute in total.

## What the scripts do

**`01_prepare.py`** builds the analysis data sets.

- It excludes the study's own app (Big4+) and assigns apps to the 3 social app categories with the codebook.
- It caps each app-day at 720 minutes. The total logged app time is the sum over all apps.
- It sums Everytime minutes separately.
- It keeps only days that also appear in the daily master file, up to 42 days from each participant's first such day.
- It builds the person-level variables:
  - the mean of ln(minutes + 1) for each social app category and for the total;
  - the mean PHQ-9 and GAD-7 scores over the 4 assessments, and whether any score was 10 or higher;
  - baseline SIAS, PSO-8, K-SAPS and GSDS scores;
  - sex, recruitment wave, age on 1 September 2023 (the covariate) and age in completed years at baseline (Table 1).

**`02_models.py`** computes the statistics and writes them to `results.json`.

- Descriptive statistics: participant flow (`flow`), sample (`sample`), Table 1 (`table1`), share of the analyzed logged time in the 3 social app categories (`app`), category use (`categories`) and anonymous community use (`community`).
- Research question 1:
  - `h1`: 6 linear models (3 categories × 2 scales) with HC3 standard errors and Benjamini-Hochberg false discovery rate control over the 6 tests;
  - `h1a`: a joint model of the 3 categories, with Wald tests of the differences between anonymous community use and the other 2 categories.
- Research question 2 (`h1c`): the anonymous community models with additional adjustment for the baseline scales.
- Research question 3:
  - `h1d_tertiles`: tertiles of anonymous community use, with Wilson CIs (the values in Figure 1);
  - `h1d_auc`: the area under the curve (AUC) of anonymous community use alone, computed as the Mann-Whitney U statistic divided by the product of the 2 group sizes (ties count one half), with a percentile bootstrap CI from 2000 resamples of participants (seed 20261008; resamples with only 1 class would be skipped, and `n_boot` gives the number used);
  - `h1d_roc`: the points of the receiver operating characteristic curves in Figure 2 (false positive and true positive rates at every cut point).
- Research question 4 (`h2*`):
  - 14-day windows before each assessment at 2, 4 and 6 weeks, with an app log on at least 7 of the 14 days;
  - linear mixed models with within- and between-participant terms for anonymous community use, with the within-participant coefficient expressed per 1 pooled within-participant SD (the square root of the sum of squared deviations from the participants' own means divided by the number of windows minus the number of participants);
  - intraclass correlations;
  - within-participant estimates on the between-participant scale;
  - a Wald test of the between-minus-within difference.

## Outputs

All outputs are written to the folder given by `--out-dir`.

| File | Content |
|---|---|
| `analysis_dataset.csv` | One row per participant-day: minutes per social app category (`min_Community`, `min_SNS`, `min_Messenger`), total minutes (`min_TOTAL`) and Everytime minutes (`min_Everytime`) |
| `person_level.csv` | One row per participant |
| `assessments.csv` | PHQ-9 and GAD-7 scores at each assessment, with dates |
| `prepare_summary.json` | Participant flow counts |
| `results.json` | All numbers reported in the article (the tertile boundaries in the Figure 1 caption are `h1d_tertile_bounds_log`, on the scale of the mean of ln(minutes + 1)) |

The first three files contain participant-level data. Do not share or commit them; `.gitignore` excludes the `data/` and `output/` folders and all CSV and Excel files.
