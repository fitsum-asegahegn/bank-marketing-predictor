import io
import joblib
import pandas as pd
import streamlit as st
import bank_features  # noqa: F401  (needed so the saved model can be loaded)

JOBS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed", "unknown"]
CHOICES = {
    "job": JOBS,
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["yes", "no"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["unknown", "failure", "other", "success"],
}
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
FEATURES = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
            "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]


@st.cache_resource
def load_bundle():
    return joblib.load("bank_model.joblib")


def predict(frame):
    bundle = load_bundle()
    proba = bundle["model"].predict_proba(frame[FEATURES])[:, 1]
    out = frame[FEATURES].copy()
    out["Probability_subscribe"] = proba.round(3)
    out["Prediction"] = ["Yes" if p >= bundle["threshold"] else "No" for p in proba]
    return out


def validate(frame):
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        return "Missing columns: " + ", ".join(missing)
    data = frame[FEATURES]
    if data.isna().any().any():
        return "The file has empty cells in the required columns."
    for col, allowed in CHOICES.items():
        bad = ~data[col].astype(str).str.lower().isin([a.lower() for a in allowed])
        if bad.any():
            return "Column %s has values that are not allowed (row %d). Allowed: %s" % (
                col, data.index[bad][0] + 2, ", ".join(allowed))
    for col in NUMERIC:
        if not pd.to_numeric(data[col], errors="coerce").notna().all():
            return "Column %s must contain numbers only." % col
    return None


st.set_page_config(page_title="Term deposit predictor", page_icon="🏦")
st.title("🏦 Term deposit subscription predictor")
st.caption("Estimates the chance that a bank client subscribes to a term deposit after a marketing "
           "call. The model does not use call duration, because it is only known after the call.")

tab1, tab2 = st.tabs(["Single client", "Upload Excel / CSV"])

with tab1:
    c1, c2 = st.columns(2)
    age = c1.number_input("age", 18, 100, 40)
    balance = c2.number_input("balance (euros, yearly average)", -10000, 120000, 1000)
    job = c1.selectbox("job", CHOICES["job"])
    marital = c2.selectbox("marital", CHOICES["marital"])
    education = c1.selectbox("education", CHOICES["education"])
    default = c2.selectbox("has credit in default?", CHOICES["default"])
    housing = c1.selectbox("has housing loan?", CHOICES["housing"])
    loan = c2.selectbox("has personal loan?", CHOICES["loan"])
    contact = c1.selectbox("contact type", CHOICES["contact"])
    month = c2.selectbox("last contact month", CHOICES["month"], index=4)
    day = c1.number_input("last contact day of month", 1, 31, 15)
    campaign = c2.number_input("contacts in this campaign", 1, 70, 1)
    never = st.checkbox("Not contacted in a previous campaign", value=True)
    pdays = -1 if never else st.number_input("days since previous campaign contact", 0, 900, 90)
    previous = c1.number_input("contacts before this campaign", 0, 60, 0)
    poutcome = c2.selectbox("outcome of previous campaign", CHOICES["poutcome"])
    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{"age": age, "job": job, "marital": marital, "education": education,
                             "default": default, "balance": balance, "housing": housing, "loan": loan,
                             "contact": contact, "day": day, "month": month, "campaign": campaign,
                             "pdays": pdays, "previous": previous, "poutcome": poutcome}])
        res = predict(row).iloc[0]
        msg = "Probability of subscribing: %.1f%%  |  prediction: %s" % (
            res["Probability_subscribe"] * 100, res["Prediction"])
        (st.success if res["Prediction"] == "Yes" else st.info)(msg)

with tab2:
    st.write("Upload a file with these columns: " + ", ".join(FEATURES))
    st.write("Use -1 in `pdays` for clients not contacted before. Use the same labels as the dataset "
             "(for example job = management, contact = cellular).")
    template = pd.DataFrame([{"age": 40, "job": "management", "marital": "married",
                              "education": "tertiary", "default": "no", "balance": 1500,
                              "housing": "yes", "loan": "no", "contact": "cellular", "day": 15,
                              "month": "may", "campaign": 1, "pdays": -1, "previous": 0,
                              "poutcome": "unknown"}])
    buf = io.BytesIO()
    template.to_excel(buf, index=False)
    st.download_button("Download template (Excel)", buf.getvalue(), "bank_template.xlsx")
    up = st.file_uploader("Excel or CSV file", type=["xlsx", "xls", "csv"])
    if up is not None:
        try:
            data = (pd.read_csv(up, sep=None, engine="python") if up.name.lower().endswith(".csv")
                    else pd.read_excel(up))
        except Exception as e:
            st.error("Could not read the file: %s" % e)
        else:
            err = validate(data)
            if err:
                st.error(err)
            else:
                data[NUMERIC] = data[NUMERIC].astype(float)
                for col in CHOICES:
                    data[col] = data[col].astype(str).str.lower()
                out = predict(data)
                st.dataframe(out)
                st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                                   "bank_predictions.csv", "text/csv")
