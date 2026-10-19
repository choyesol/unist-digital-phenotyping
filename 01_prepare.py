import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["Community", "SNS", "Messenger"]
EVERYTIME = ["Everytime", "에브리타임"]
STUDY_APP = "Big4+"
APP_CAP_MIN = 720
TRIM_DAYS = 42
BASELINE_SCALES = ["SIAS", "PSO", "Smartphone", "GSDS"]


def norm_name(s):
    s = "".join(ch for ch in str(s) if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", s).strip().casefold()


def survey_date(v):
    s = str(v)
    return pd.to_datetime("20" + s if len(s) == 6 else "2023" + s.zfill(4), format="%Y%m%d")


def daily_app_minutes(log_paths, codebook):
    app = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in log_paths], ignore_index=True)
    log_users = set(app.userId)
    app["date"] = pd.to_datetime(app.appLogRecordDate).dt.normalize() - pd.Timedelta(days=1)
    app["key"] = app.appName.map(norm_name)
    app = app[app.key != norm_name(STUDY_APP)]

    rows = []
    for (user, key), g in app.groupby(["userId", "key"], sort=False):
        previous = set()
        for day, d in g.groupby("date"):
            values = set(d.appRunTimeForeground)
            new = values - previous
            rows.append((user, day, key, max(new) if new else 0))
            previous = values
    per_app = pd.DataFrame(rows, columns=["userId", "date", "key", "sec"])

    book = pd.read_csv(codebook, encoding="utf-8-sig")
    category = dict(zip(book.appName.map(norm_name), book.appCategory))
    per_app["category"] = per_app.key.map(category).fillna("Other")
    per_app["sec"] = per_app.sec.clip(upper=APP_CAP_MIN * 60)

    daily = per_app.groupby(["userId", "date", "category"]).sec.sum().unstack(fill_value=0)
    daily = daily.reindex(columns=CATEGORIES, fill_value=0).div(60).add_prefix("min_")
    daily["min_TOTAL"] = per_app.groupby(["userId", "date"]).sec.sum().div(60).reindex(daily.index)
    everytime = per_app[per_app.key.isin({norm_name(n) for n in EVERYTIME})]
    daily["min_Everytime"] = everytime.groupby(["userId", "date"]).sec.sum().div(60).reindex(daily.index, fill_value=0)
    return daily.reset_index(), log_users


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--app-logs", nargs="+", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    daily, log_users = daily_app_minutes([args.data_dir / f for f in args.app_logs], args.codebook)
    master = pd.read_excel(args.data_dir / args.master, sheet_name="Data", usecols=["userId", "deviceType", "Date"])
    survey = pd.read_excel(args.data_dir / args.survey)
    survey = survey.rename(columns={"ID": "userId", "Week": "week", "연구": "wave"})

    ds = master[["userId", "Date"]].rename(columns={"Date": "date"}).merge(daily, on=["userId", "date"])
    ds = ds[ds.date < ds.groupby("userId").date.transform("min") + pd.Timedelta(days=TRIM_DAYS)]
    ds = ds.sort_values(["userId", "date"]).reset_index(drop=True)
    ds.to_csv(args.out_dir / "analysis_dataset.csv", index=False)

    users = sorted(ds.userId.unique())
    s = survey[survey.userId.isin(users)].assign(date=lambda x: x.Date.map(survey_date))
    s[["userId", "week", "date", "PHQ", "GAD"]].to_csv(args.out_dir / "assessments.csv", index=False)

    P = pd.DataFrame(index=pd.Index(users, name="userId"))
    for c in CATEGORIES + ["TOTAL"]:
        P["use_" + c] = np.log1p(ds["min_" + c]).groupby(ds.userId).mean()
    for sc in ("PHQ", "GAD"):
        scores = s.pivot(index="userId", columns="week", values=sc)
        P[sc + "_mean4"] = scores.mean(axis=1)
        P[sc + "_any10"] = (scores >= 10).any(axis=1).astype(int)
    base = s[s.week == 0].set_index("userId")
    for c in BASELINE_SCALES:
        P[c + "_w0"] = base[c]
    birth = pd.to_datetime(base.Age)
    P["female"] = (base.Sex == "여").astype(int)
    P["wave2"] = base.wave.str.contains("2").astype(int)
    P["age"] = (pd.Timestamp("2023-09-01") - birth).dt.days / 365.25
    P["age_baseline"] = np.floor((base.date - birth).dt.days / 365.25)
    P.to_csv(args.out_dir / "person_level.csv")

    parent = set(survey.userId)
    android = parent & set(master.userId[master.deviceType == "Android"])
    summary = dict(parent=len(parent), android=len(android), android_with_any_log=len(android & log_users))
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
