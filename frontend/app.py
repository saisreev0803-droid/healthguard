import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import duckdb

st.set_page_config(
    page_title="HealthGuard",
    page_icon="🏥",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def call_predict(features: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        return response.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def get_high_risk_patients():
    try:
        response = requests.get(f"{API_URL}/high-risk-patients?limit=20", timeout=10)
        return response.json()
    except:
        return []

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def make_gauge(risk_score: float, risk_label: str):
    color = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(risk_score * 100, 1),
        title={"text": f"Readmission Risk<br><span style='font-size:0.8em'>{risk_label}</span>"},
        delta={"reference": 20, "suffix": "%"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color.get(risk_label, "#3498db")},
            "steps": [
                {"range": [0, 35],  "color": "#eafaf1"},
                {"range": [35, 65], "color": "#fef9e7"},
                {"range": [65, 100],"color": "#fdedec"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": risk_score * 100,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=0, l=20, r=20))
    return fig

def make_shap_chart(top_risk_factors: list):
    names   = [f["feature"] for f in top_risk_factors]
    impacts = [f["impact"]  for f in top_risk_factors]
    colors  = ["#e74c3c" if i > 0 else "#2ecc71" for i in impacts]
    fig = go.Figure(go.Bar(
        x=impacts,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{'+' if i > 0 else ''}{i:.3f}" for i in impacts],
        textposition="outside",
    ))
    fig.update_layout(
        title="Why did the model give this score?",
        xaxis_title="Impact on risk score",
        height=350,
        margin=dict(t=50, b=40, l=20, r=60),
        yaxis=dict(autorange="reversed"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

st.sidebar.title("🏥 HealthGuard")
st.sidebar.caption("AI-Powered Readmission Risk")

page = st.sidebar.radio(
    "Navigate",
    ["Patient Risk Check", "High Risk Patients", "Population Analytics"],
)

api_ok = check_api()
if api_ok:
    st.sidebar.success("API: Online")
else:
    st.sidebar.error("API: Offline — start uvicorn first")

st.sidebar.markdown("---")
st.sidebar.caption("HealthGuard v1.0 · Built with FastAPI + XGBoost + Streamlit")

# ─────────────────────────────────────────────────────────────────
# PAGE 1 — Patient Risk Check
# ─────────────────────────────────────────────────────────────────

if page == "Patient Risk Check":

    st.title("🏥 Patient Readmission Risk Check")
    st.caption("Enter patient clinical details to get a 30-day readmission risk prediction.")
    st.markdown("---")

    with st.form("patient_form"):

        st.subheader("Patient Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=110, value=67)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            gender_encoded = 0 if gender == "Male" else 1
        with col3:
            insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Private", "Uninsured"])
            insurance_map = {"Medicare": 0, "Medicaid": 1, "Private": 2, "Uninsured": 3}
            insurance_encoded = insurance_map[insurance]

        st.markdown("---")
        st.subheader("Hospital History")
        col4, col5, col6 = st.columns(3)
        with col4:
            num_admissions = st.number_input("Admissions (last 12 months)", 0, 20, value=3)
        with col5:
            total_los = st.number_input("Total days in hospital", 0, 365, value=18)
        with col6:
            max_stay = st.number_input("Longest single stay (days)", 0, 180, value=9)

        avg_los = round(total_los / max(num_admissions, 1), 1)
        st.caption(f"Average length of stay: **{avg_los} days**")

        st.markdown("---")
        st.subheader("Diagnoses")
        col7, col8 = st.columns(2)
        with col7:
            num_chronic      = st.slider("Number of chronic conditions", 0, 15, value=4)
            has_diabetes     = int(st.checkbox("Type 2 Diabetes",       value=True))
            has_heart_failure= int(st.checkbox("Heart Failure",          value=True))
        with col8:
            has_ckd          = int(st.checkbox("Chronic Kidney Disease", value=False))
            has_hypertension = int(st.checkbox("Hypertension",           value=True))
            has_copd         = int(st.checkbox("COPD",                   value=False))

        st.markdown("---")
        st.subheader("Medications")
        col9, col10 = st.columns(2)
        with col9:
            num_meds = st.number_input("Total medications", 0, 30, value=7)
        with col10:
            num_active_meds = st.number_input("Active medications", 0, 30, value=5)

        st.markdown("---")
        st.subheader("Latest Lab Results")
        col11, col12, col13 = st.columns(3)
        with col11:
            avg_glucose    = st.number_input("Blood Glucose (mg/dL)", 0.0, 600.0, value=210.5)
            avg_hba1c      = st.number_input("HbA1c (%)",             0.0, 20.0,  value=8.2)
        with col12:
            avg_systolic   = st.number_input("Systolic BP (mmHg)",    0.0, 300.0, value=148.0)
            avg_heart_rate = st.number_input("Heart Rate (bpm)",      0.0, 250.0, value=88.0)
        with col13:
            avg_creatinine = st.number_input("Creatinine (mg/dL)",    0.0, 20.0,  value=1.8)

        submitted = st.form_submit_button(
            "Predict Readmission Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not api_ok:
            st.error("API is offline. Please start the uvicorn server first.")
        else:
            with st.spinner("Running prediction..."):
                features = {
                    "age":                    float(age),
                    "gender_encoded":         gender_encoded,
                    "insurance_encoded":      insurance_encoded,
                    "num_admissions_12m":     float(num_admissions),
                    "total_length_of_stay":   float(total_los),
                    "max_single_stay":        float(max_stay),
                    "avg_length_of_stay":     float(avg_los),
                    "num_chronic_conditions": float(num_chronic),
                    "has_diabetes":           has_diabetes,
                    "has_heart_failure":      has_heart_failure,
                    "has_ckd":                has_ckd,
                    "has_hypertension":       has_hypertension,
                    "has_copd":               has_copd,
                    "num_medications":        float(num_meds),
                    "num_active_medications": float(num_active_meds),
                    "avg_glucose":            avg_glucose,
                    "avg_systolic_bp":        avg_systolic,
                    "avg_creatinine":         avg_creatinine,
                    "avg_hba1c":              avg_hba1c,
                    "avg_heart_rate":         avg_heart_rate,
                }
                result = call_predict(features)

            if result:
                st.markdown("---")
                st.subheader("Prediction Results")

                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Score",  result["risk_percent"])
                m2.metric("Risk Level",  result["risk_label"])
                m3.metric("Baseline",    "~20% avg population risk")

                g_col, s_col = st.columns(2)
                with g_col:
                    st.plotly_chart(
                        make_gauge(result["risk_score"], result["risk_label"]),
                        use_container_width=True
                    )
                with s_col:
                    st.plotly_chart(
                        make_shap_chart(result["top_risk_factors"]),
                        use_container_width=True
                    )

                label = result["risk_label"]
                if label == "High Risk":
                    st.error(f"**Recommendation:** {result['recommendation']}")
                elif label == "Medium Risk":
                    st.warning(f"**Recommendation:** {result['recommendation']}")
                else:
                    st.success(f"**Recommendation:** {result['recommendation']}")

                st.markdown("---")
                st.subheader("Full Risk Factor Breakdown")
                factors_df = pd.DataFrame(result["top_risk_factors"])
                factors_df.columns = ["Feature", "Patient Value", "Risk Impact", "Direction"]
                st.dataframe(factors_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE 2 — High Risk Patients
# ─────────────────────────────────────────────────────────────────

elif page == "High Risk Patients":

    st.title("⚠️ High Risk Patient List")
    st.caption("Patients ranked by admission history and chronic condition burden.")
    st.markdown("---")

    with st.spinner("Loading patients..."):
        patients = get_high_risk_patients()

    if patients:
        df = pd.DataFrame(patients)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients Shown",         len(df))
        c2.metric("Avg Admissions",         round(df["num_admissions_12m"].mean(), 1))
        c3.metric("Avg Chronic Conditions", round(df["num_chronic_conditions"].mean(), 1))
        c4.metric("Heart Failure Cases",    int(df["has_heart_failure"].sum()))

        st.markdown("---")
        st.dataframe(df, use_container_width=True, height=500)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="high_risk_patients.csv",
            mime="text/csv",
        )
    else:
        st.warning("Could not load patients. Make sure the API is running.")

# ─────────────────────────────────────────────────────────────────
# PAGE 3 — Population Analytics
# ─────────────────────────────────────────────────────────────────

elif page == "Population Analytics":

    st.title("📊 Population Analytics")
    st.caption("Overview of the full patient population in the database.")
    st.markdown("---")

    try:
        con = duckdb.connect("data/health.db")

        totals = con.execute("""
            SELECT
                COUNT(*)                                AS total_patients,
                ROUND(AVG(p.age), 1)                    AS avg_age,
                SUM(f.readmitted_30days)                AS total_readmitted,
                ROUND(AVG(f.num_chronic_conditions), 1) AS avg_conditions
            FROM features f
            JOIN patients p ON f.patient_id = p.patient_id
        """).df().iloc[0]

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Patients",         int(totals["total_patients"]))
        t2.metric("Average Age",            totals["avg_age"])
        t3.metric("Readmitted (30 days)",   int(totals["total_readmitted"]))
        t4.metric("Avg Chronic Conditions", totals["avg_conditions"])

        st.markdown("---")
        row1_left, row1_right = st.columns(2)

        with row1_left:
            age_data = con.execute("SELECT p.age FROM patients p").df()
            fig_age = px.histogram(
                age_data, x="age", nbins=20,
                title="Patient Age Distribution",
                color_discrete_sequence=["#3498db"],
            )
            fig_age.update_layout(height=350)
            st.plotly_chart(fig_age, use_container_width=True)

        with row1_right:
            ins_data = con.execute("""
                SELECT insurance, COUNT(*) AS count
                FROM patients
                GROUP BY insurance
            """).df()
            fig_ins = px.pie(
                ins_data, names="insurance", values="count",
                title="Insurance Type Breakdown",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_ins.update_layout(height=350)
            st.plotly_chart(fig_ins, use_container_width=True)

        st.markdown("---")
        row2_left, row2_right = st.columns(2)

        with row2_left:
            diag_data = con.execute("""
                SELECT diagnosis, COUNT(*) AS total
                FROM conditions
                GROUP BY diagnosis
                ORDER BY total DESC
                LIMIT 10
            """).df()
            fig_diag = px.bar(
                diag_data, x="total", y="diagnosis",
                orientation="h",
                title="Top 10 Most Common Diagnoses",
                color="total",
                color_continuous_scale="Blues",
                labels={"total": "Patients", "diagnosis": ""},
            )
            fig_diag.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_diag, use_container_width=True)

        with row2_right:
            dept_data = con.execute("""
                SELECT
                    department,
                    COUNT(encounter_id)           AS total_visits,
                    ROUND(AVG(length_of_stay), 1) AS avg_los
                FROM encounters
                GROUP BY department
                ORDER BY avg_los DESC
            """).df()
            fig_dept = px.bar(
                dept_data, x="department", y="avg_los",
                title="Average Length of Stay by Department",
                color="avg_los",
                color_continuous_scale="Reds",
                labels={"avg_los": "Avg Days", "department": "Department"},
            )
            fig_dept.update_layout(height=400)
            st.plotly_chart(fig_dept, use_container_width=True)

        con.close()

    except Exception as e:
        st.error(f"Could not load analytics: {e}")
        st.caption("Make sure data/health.db exists and scripts have been run.")