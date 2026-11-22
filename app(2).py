import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import bank_features  # noqa: F401  (required to unpickle the model)
from bank_features import RAW_COLS, NUM_COLS, MONTHS

HERE = Path(__file__).parent
CHOICES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["yes", "no"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": MONTHS,
    "poutcome": ["unknown", "failure", "other", "success"],
}
LIMITS = {"age": (18, 100), "balance": (-100000, 1000000), "day": (1, 31), "campaign": (1, 100), "previous": (0, 500)}
TEMPLATE = pd.DataFrame([
    {"age": 40, "job": "management", "marital": "married", "education": "tertiary", "default": "no", "balance": 1500,
     "housing": "yes", "loan": "no", "contact": "cellular", "day": 15, "month": "may", "campaign": 1, "pdays": -1,
     "previous": 0, "poutcome": "unknown"},
    {"age": 29, "job": "student", "marital": "single", "education": "secondary", "default": "no", "balance": 800,
     "housing": "no", "loan": "no", "contact": "cellular", "day": 8, "month": "sep", "campaign": 2, "pdays": 120,
     "previous": 2, "poutcome": "success"},
])


@st.cache_resource
def load_bundle():
    return joblib.load(HERE / "bank_model.joblib")


@st.cache_data
def load_report():
    path = HERE / "model_report.json"
    return json.loads(path.read_text()) if path.exists() else None


def band(p, bundle):
    if p >= bundle["threshold"]:
        return "High"
    return "Medium" if p >= bundle["base_rate"] else "Low"


def score(frame):
    bundle = load_bundle()
    p = bundle["model"].predict_proba(frame[RAW_COLS])[:, 1]
    out = frame[RAW_COLS].copy()
    out.insert(0, "probability", p.round(4))
    out.insert(1, "priority", [band(v, bundle) for v in p])
    out.insert(2, "decision", ["Call" if v >= bundle["threshold"] else "Skip" for v in p])
    return out


def clean_upload(raw):
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        return None, ["Missing columns: " + ", ".join(missing)]
    data = raw[RAW_COLS].copy()
    errors = []
    empty = data.isna().any(axis=1)
    if empty.any():
        errors.append("Empty cells in rows: " + ", ".join(map(str, (data.index[empty] + 2)[:10])))
        data = data[~empty]
    for col in CHOICES:
        data[col] = data[col].astype(str).str.strip().str.lower()
        bad = ~data[col].isin(CHOICES[col])
        if bad.any():
            errors.append(f"{col}: unsupported value in rows {', '.join(map(str, (data.index[bad] + 2)[:10]))}. "
                          f"Allowed: {', '.join(CHOICES[col])}")
    for col in NUM_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().any():
            errors.append(f"{col}: must be numeric (rows {', '.join(map(str, (data.index[data[col].isna()] + 2)[:10]))})")
    for col, (lo, hi) in LIMITS.items():
        if data[col].isna().any():
            continue
        out = ~data[col].between(lo, hi)
        if out.any():
            errors.append(f"{col}: must be between {lo} and {hi} (rows {', '.join(map(str, (data.index[out] + 2)[:10]))})")
    bad_p = data["pdays"].notna() & ~((data["pdays"] == -1) | (data["pdays"] >= 1))
    if bad_p.any():
        errors.append("pdays: use -1 for never contacted, otherwise 1 or more days")
    return (None, errors) if errors else (data.reset_index(drop=True), [])


def to_excel(frame):
    buf = io.BytesIO()
    frame.to_excel(buf, index=False)
    return buf.getvalue()


st.set_page_config(page_title="Term deposit predictor", page_icon="🏦", layout="wide")

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load bank_model.joblib: {exc}")
    st.stop()
report = load_report()

st.title("🏦 Term deposit subscription predictor")
st.caption("Ranks bank clients by the probability of subscribing to a term deposit after a marketing call. "
           "Uses only information known before the call (no call duration).")

with st.sidebar:
    st.subheader("Model in use")
    st.write(bundle["name"])
    st.metric("Call threshold", f"{bundle['threshold']:.0%}")
    st.metric("Average subscription rate", f"{bundle['base_rate']:.1%}")
    if report:
        st.metric("Test ROC-AUC", f"{report['test']['ROC-AUC']['estimate']:.3f}")
        st.metric("Test recall at threshold", f"{report['test']['Recall']['estimate']:.0%}")
        st.metric("Test precision at threshold", f"{report['test']['Precision']['estimate']:.0%}")

tab1, tab2, tab3 = st.tabs(["Single client", "Excel / CSV upload", "Model report"])

with tab1:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, 40)
        job = c2.selectbox("Job", CHOICES["job"], index=4)
        marital = c3.selectbox("Marital status", CHOICES["marital"])
        education = c1.selectbox("Education", CHOICES["education"], index=1)
        default = c2.selectbox("Credit in default", CHOICES["default"])
        balance = c3.number_input("Average yearly balance (euros)", -100000, 1000000, 1000, step=100)
        housing = c1.selectbox("Housing loan", CHOICES["housing"])
        loan = c2.selectbox("Personal loan", CHOICES["loan"])
        contact = c3.selectbox("Contact type", CHOICES["contact"])
        month = c1.selectbox("Last contact month", CHOICES["month"], index=4)
        day = c2.number_input("Last contact day of month", 1, 31, 15)
        campaign = c3.number_input("Contacts during this campaign", 1, 100, 1)
        previous = c1.number_input("Contacts before this campaign", 0, 500, 0)
        poutcome = c2.selectbox("Outcome of previous campaign", CHOICES["poutcome"])
        pdays = c3.number_input("Days since previous contact (-1 if never)", -1, 900, -1)
        go = st.form_submit_button("Predict", type="primary")
    if go:
        if pdays != -1 and pdays < 1:
            st.error("Days since previous contact must be -1 (never) or at least 1.")
        else:
            row = pd.DataFrame([{"age": age, "job": job, "marital": marital, "education": education,
                                 "default": default, "balance": balance, "housing": housing, "loan": loan,
                                 "contact": contact, "day": day, "month": month, "campaign": campaign,
                                 "pdays": pdays, "previous": previous, "poutcome": poutcome}])
            res = score(row).iloc[0]
            p = float(res["probability"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability of subscribing", f"{p:.1%}")
            m2.metric("Priority", res["priority"])
            m3.metric("Decision", res["decision"], f"{p / bundle['base_rate']:.1f}x average rate", delta_color="off")
            st.progress(min(p, 1.0))
            msg = (f"Call this client: probability {p:.1%} is above the {bundle['threshold']:.0%} threshold."
                   if res["decision"] == "Call" else
                   f"Lower priority: probability {p:.1%} is below the {bundle['threshold']:.0%} threshold.")
            (st.success if res["decision"] == "Call" else st.info)(msg)

with tab2:
    st.write("Required columns: " + ", ".join(f"`{c}`" for c in RAW_COLS))
    st.write("Use `-1` in `pdays` for clients never contacted before. Labels follow the original dataset "
             "(for example `job = management`, `contact = cellular`).")
    st.download_button("Download Excel template", to_excel(TEMPLATE), "bank_template.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    up = st.file_uploader("Excel or CSV file", type=["xlsx", "xls", "csv"])
    if up is not None:
        try:
            raw = (pd.read_csv(up, sep=None, engine="python") if up.name.lower().endswith(".csv")
                   else pd.read_excel(up))
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
        else:
            data, errors = clean_upload(raw)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                out = score(data).sort_values("probability", ascending=False).reset_index(drop=True)
                n_call = int((out["decision"] == "Call").sum())
                a, b, c = st.columns(3)
                a.metric("Clients scored", f"{len(out):,}")
                b.metric("Recommended to call", f"{n_call:,}", f"{n_call / len(out):.0%} of the list", delta_color="off")
                c.metric("Expected subscribers (sum of probabilities)", f"{out['probability'].sum():.0f}")
                st.dataframe(out, use_container_width=True, height=420)
                d1, d2 = st.columns(2)
                d1.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                                   "bank_predictions.csv", "text/csv")
                d2.download_button("Download predictions (Excel)", to_excel(out), "bank_predictions.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab3:
    if not report:
        st.info("model_report.json not found next to app.py.")
    else:
        t = report["test"]
        st.subheader("Held-out test results")
        cols = st.columns(5)
        for col, k in zip(cols, ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]):
            col.metric(k, f"{t[k]['estimate']:.3f}", f"95% CI {t[k]['lo']:.3f} to {t[k]['hi']:.3f}", delta_color="off")
        cm = report["confusion"]
        left, right = st.columns(2)
        with left:
            st.write("Confusion matrix (threshold %.2f)" % report["threshold"])
            st.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                                      index=["actual no", "actual yes"], columns=["predicted no", "predicted yes"]))
            called = cm["tp"] + cm["fp"]
            st.write(f"The model flags {called:,} of {sum(cm.values()):,} clients and reaches "
                     f"{cm['tp'] / (cm['tp'] + cm['fn']):.0%} of all subscribers, with {cm['tp'] / called:.0%} of calls successful "
                     f"(average rate {report['base_rate']:.1%}).")
        with right:
            st.write("Subscription rate by predicted-probability decile (out-of-fold, training data)")
            lift = pd.DataFrame(report["lift"]).set_index("decile")
            st.bar_chart(lift["rate"])
        st.subheader("Model comparison (10-fold cross-validation on the training set)")
        comp = pd.DataFrame(report["comparison"]).set_index("model")
        st.dataframe(comp.style.format(precision=3).highlight_max(subset=["ROC-AUC", "PR-AUC", "Lift@10%", "Recall@20%"], color="#cfe8cf")
                     .highlight_min(subset=["Brier"], color="#cfe8cf"), use_container_width=True)
        l2, r2 = st.columns(2)
        with l2:
            st.write("Permutation importance")
            st.bar_chart(pd.Series(report["importance"]))
        with r2:
            st.write("ROC curve (test)")
            st.line_chart(pd.DataFrame({"TPR": report["roc"]["tpr"]}, index=report["roc"]["fpr"]), x_label="FPR", y_label="TPR")
