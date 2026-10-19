import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

CATEGORIES = ["Community", "SNS", "Messenger"]
OUTCOMES = ["PHQ_mean4", "GAD_mean4"]
COVARIATES = "ztot + female + wave2 + age"
ADJUSTMENTS = ["SIAS_w0", "PSO_w0", "Smartphone_w0", "GSDS_w0", "SIAS_w0 + PSO_w0 + Smartphone_w0 + GSDS_w0"]
SEED = 20261008
N_BOOT = 2000
WINDOW_DAYS = 14
MIN_WINDOW_DAYS = 7


def z(x):
    return (x - x.mean()) / x.std()


def msd(x):
    return dict(mean=x.mean(), sd=x.std())


def mediqr(x):
    return dict(median=x.median(), q1=x.quantile(0.25), q3=x.quantile(0.75))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def ols(formula, data, term):
    m = smf.ols(formula, data).fit(cov_type="HC3")
    lo, hi = m.conf_int().loc[term]
    return dict(b=m.params[term], lo=lo, hi=hi, p=m.pvalues[term])


def wilson(k, n, zc=1.96):
    p = k / n
    den = 1 + zc * zc / n
    c = (p + zc * zc / (2 * n)) / den
    h = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / den
    return c - h, c + h


def auc(d, y):
    case = d[y] == 1
    u = stats.mannwhitneyu(d.use_Community[case], d.use_Community[~case]).statistic
    return u / (case.sum() * (~case).sum())


def descriptives(P, ds, A, prep, mins):
    days = ds.groupby("userId").size()
    scores = {sc: A.pivot(index="userId", columns="week", values=sc) for sc in ("PHQ", "GAD")}
    return dict(
        flow=dict(parent=prep["parent"], android=prep["android"], android_with_any_log=prep["android_with_any_log"],
                  analyzed=len(P)),
        sample=dict(n=len(P), wave2=int(P.wave2.sum()), female=int(P.female.sum()),
                    age=dict(msd(P.age_baseline), min=P.age_baseline.min(), max=P.age_baseline.max()),
                    app_days=dict(mediqr(days), min=days.min(), max=days.max()),
                    complete=int((A.groupby("userId")[["PHQ", "GAD"]].count() == 4).all(axis=1).sum()),
                    baseline_complete=int(P.filter(like="_w0").notna().all(axis=1).sum())),
        table1=dict(
            phq_mean4=msd(P.PHQ_mean4), gad_mean4=msd(P.GAD_mean4),
            phq_ge10_any=int(P.PHQ_any10.sum()), gad_ge10_any=int(P.GAD_any10.sum()),
            ksaps=msd(P.Smartphone_w0), gsds=msd(P.GSDS_w0), sias=msd(P.SIAS_w0), pso=msd(P.PSO_w0),
            total_min=mediqr(mins.min_TOTAL),
            rise5=dict(PHQ=int((scores["PHQ"][[2, 4, 6]].diff(axis=1).iloc[:, 1:] >= 5).any(axis=1).sum()))),
        app=dict(social_pct=100 * ds[["min_" + c for c in CATEGORIES]].sum().sum() / ds.min_TOTAL.sum()),
        categories={c: dict(minutes=mediqr(mins["min_" + c]), sd_log=P["use_" + c].std()) for c in CATEGORIES},
        community=dict(users=int((mins.min_Community > 0).sum()),
                       everytime_users=int((ds.groupby("userId").min_Everytime.sum() > 0).sum()),
                       everytime_share_pct=100 * ds.min_Everytime.sum() / ds.min_Community.sum()))


def rq1_categories(P):
    h1 = {}
    for y in OUTCOMES:
        for c in CATEGORIES:
            h1[f"{y}|{c}"] = ols(f"{y} ~ zx + {COVARIATES}", P.assign(zx=z(P["use_" + c])), "zx")
    for r, q in zip(h1.values(), bh([r["p"] for r in h1.values()])):
        r["q"] = q
    h1a = {}
    for y in OUTCOMES:
        m = smf.ols(f"{y} ~ zcom + z_SNS + z_Messenger + {COVARIATES}", P).fit(cov_type="HC3")
        ci = m.conf_int()
        h1a[y] = {k: dict(b=m.params[v], lo=ci.loc[v, 0], hi=ci.loc[v, 1], p=m.pvalues[v])
                  for k, v in (("community", "zcom"), ("sns", "z_SNS"), ("messenger", "z_Messenger"))}
        for k, v in (("sns", "z_SNS"), ("messenger", "z_Messenger")):
            t = m.t_test(f"zcom - {v} = 0")
            lo, hi = t.conf_int()[0]
            h1a[y]["diff_vs_" + k] = dict(diff=t.effect.item(), lo=lo, hi=hi, p=t.pvalue.item())
    return h1, h1a


def rq2_adjustment(P):
    return {f"{y}|{extra}": ols(f"{y} ~ zcom + {COVARIATES} + {extra}", P, "zcom") for y in OUTCOMES for extra in ADJUSTMENTS}


def rq3_magnitude(P, mins):
    tertile, edges = pd.qcut(P.use_Community, 3, labels=["low", "mid", "high"], retbins=True)
    tertiles = {}
    for t, g in P.groupby(tertile, observed=True):
        tertiles[t] = dict(n=len(g), median_min=mins.min_Community[g.index].median())
        for y in ("PHQ_any10", "GAD_any10"):
            k = int(g[y].sum())
            lo, hi = wilson(k, len(g))
            tertiles[t][y] = dict(k=k, pct=100 * k / len(g), lo=100 * lo, hi=100 * hi)

    rng = np.random.default_rng(SEED)
    aucs = {}
    for y in ("PHQ_any10", "GAD_any10"):
        boots = []
        for _ in range(N_BOOT):
            sample = P.sample(len(P), replace=True, random_state=int(rng.integers(1e9)))
            if sample[y].nunique() == 2:  # resamples with only 1 class are skipped
                boots.append(auc(sample, y))
        aucs[y] = dict(auc=auc(P, y), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5), n_boot=len(boots))
    roc = {}
    for y in ("PHQ_any10", "GAD_any10"):  # empirical ROC curve of use alone (Figure 2)
        x, cases = P.use_Community.to_numpy(), P[y].to_numpy()
        cuts = np.r_[np.inf, np.sort(np.unique(x))[::-1]]
        roc[y] = dict(fpr=[float(((x >= c) & (cases == 0)).sum() / (cases == 0).sum()) for c in cuts],
                      tpr=[float(((x >= c) & (cases == 1)).sum() / (cases == 1).sum()) for c in cuts])
    return dict(h1d_tertile_bounds_log=list(edges[1:3]), h1d_tertiles=tertiles, h1d_auc=aucs, h1d_roc=roc)


def rq4_within_person(P, ds, A):
    windows = []
    A = A[A.week.isin([2, 4, 6])]
    for a in A.itertuples():
        g = ds[(ds.userId == a.userId) & (ds.date < a.date) & (ds.date >= a.date - pd.Timedelta(days=WINDOW_DAYS))]
        if len(g) >= MIN_WINDOW_DAYS:
            windows.append(dict(userId=a.userId, week=a.week, PHQ=a.PHQ, GAD=a.GAD,
                                u_Community=np.log1p(g.min_Community).mean()))
    W = pd.DataFrame(windows).sort_values(["userId", "week"]).reset_index(drop=True)
    W = W.join(P[["wave2", "ztot", "female", "age"]], on="userId")

    icc = {}
    for v in ("PHQ", "GAD", "u_Community"):
        m = smf.mixedlm(f"{v} ~ 1", W, groups=W.userId).fit(reml=True)
        icc[v] = m.cov_re.iloc[0, 0] / (m.cov_re.iloc[0, 0] + m.scale)

    sdb = P.use_Community.std()
    d = W.assign(ub=W.groupby("userId").u_Community.transform("mean"))
    d["uw"] = d.u_Community - d.ub
    sdw = np.sqrt((d.uw ** 2).sum() / (len(d) - d.userId.nunique()))  # pooled within-participant SD
    h2, same, compare = {}, {}, {}
    for sc in ("PHQ", "GAD"):
        m = smf.mixedlm(f"{sc} ~ uw + ub + wave2 + C(week) + ztot + female + age", d, groups=d.userId).fit(reml=True)
        ci = m.conf_int()
        b, (lo, hi) = m.params.uw * sdw, ci.loc["uw"] * sdw
        h2[f"{sc}|Community"] = dict(b=b, lo=lo, hi=hi, p=m.pvalues.uw)
        same[sc] = dict(b=b * (sdb / sdw), lo=lo * (sdb / sdw), hi=hi * (sdb / sdw))
        contrast = (m.fe_params.index == "ub").astype(float) - (m.fe_params.index == "uw")
        t = m.t_test(contrast[None, :])
        dlo, dhi = t.conf_int()[0]
        compare[sc] = dict(between=m.params.ub * sdb, between_lo=ci.loc["ub", 0] * sdb, between_hi=ci.loc["ub", 1] * sdb,
                           diff=t.effect.item() * sdb, diff_lo=dlo * sdb, diff_hi=dhi * sdb, diff_p=t.pvalue.item())
    summary = dict(windows=len(d), persons=d.userId.nunique(), candidate=len(A))
    return dict(h2=h2, h2_summary=summary, h2_community_same_scale=same, h2_community_compare=compare, h2_icc=icc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    out = parser.parse_args().out_dir

    ds = pd.read_csv(out / "analysis_dataset.csv", parse_dates=["date"])
    P = pd.read_csv(out / "person_level.csv", index_col="userId")
    A = pd.read_csv(out / "assessments.csv", parse_dates=["date"])
    prep = json.loads((out / "prepare_summary.json").read_text())
    P["ztot"] = z(P.use_TOTAL)
    P["zcom"] = z(P.use_Community)
    P["z_SNS"] = z(P.use_SNS)
    P["z_Messenger"] = z(P.use_Messenger)
    mins = ds.groupby("userId")[["min_" + c for c in CATEGORIES + ["TOTAL"]]].mean()

    R = descriptives(P, ds, A, prep, mins)
    R["h1"], R["h1a"] = rq1_categories(P)
    R["h1c"] = rq2_adjustment(P)
    R.update(rq3_magnitude(P, mins))
    R.update(rq4_within_person(P, ds, A))
    (out / "results.json").write_text(json.dumps(R, indent=1, default=lambda v: v.item()))


if __name__ == "__main__":
    main()
