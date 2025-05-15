import streamlit as st
import pandas as pd
import requests
import json
import os
from datetime import datetime, date

# API_ENDPOINT = "http://localhost:8000/transactions/"
# UPLOAD_ENDPOINT = "http://localhost:8000/upload-csv/"
API_ENDPOINT = "http://10.255.78.58:9004/transactions/"
UPLOAD_ENDPOINT = "http://10.255.78.58:9004/upload-csv/"
STATIC_DIR = "static"

st.title("Anomaly Detection System")

# Section 1: Data Source Selection
st.header("📤 Compare Aggregation Performance")
data_source = st.radio("📁 Select Data Source", ["Upload CSV File", "Use Static File"], index=0)

uploaded_df = None
file_to_upload = None

# Option 1: User uploads a file
if data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV with 100 Transactions", type="csv")
    upload_btn_disabled = False
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df.columns = uploaded_df.columns.str.strip()
        uploaded_df = uploaded_df.dropna(how='all')

        st.success("✅ File uploaded!")
        st.markdown("### 📄 Uploaded CSV Preview")
        st.dataframe(uploaded_df)

        uploaded_file.seek(0)  # Reset file pointer

        if st.button("🚀 Upload and Insert All", disabled=upload_btn_disabled):
            with st.spinner("Uploading and analyzing transactions..."):
                upload_btn_disabled = True  # Optional flag
                try:
                    response = requests.post(UPLOAD_ENDPOINT, params={"aggregation_mode": "table"}, files={"file": uploaded_file})
                    if response.ok:
                        json_response = response.json()
                        # ... existing logic to display abnormal records and graphs ...
                        if "abnormal_records" in json_response:
                            df_abnormal = pd.DataFrame(json_response["abnormal_records"])
                            if not df_abnormal.empty:
                                st.markdown("### ⚠️ Abnormal Records Detected")
                                st.dataframe(df_abnormal)

                                csv = df_abnormal.to_csv(index=False).encode("utf-8")
                                st.download_button("📥 Download Abnormal Records", csv, "abnormal_records.csv", "text/csv")

                            if uploaded_df is not None:
                                df_uploaded = uploaded_df.copy()
                                df_uploaded["Transaction Date"] = df_uploaded["Transaction Date"].astype(str)

                                total_records = len(df_uploaded)
                                abnormal_records = len(df_abnormal)
                                normal_records = total_records - abnormal_records

                                df_overall = pd.DataFrame({
                                    "Transaction Type": ["Abnormal", "Normal"],
                                    "Count": [abnormal_records, normal_records]
                                })

                                st.markdown("### 📊 Overall Abnormal vs Normal Transactions")
                                st.bar_chart(df_overall.set_index("Transaction Type"))

                        else:
                            st.success("✅ No abnormal transactions detected.")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")

# Option 2: User selects from static folder
else:
    available_files = [f for f in os.listdir(STATIC_DIR) if f.endswith(".csv")]
    static_file_name = st.selectbox("📂 Choose Static CSV File", available_files)
    if static_file_name:
        file_path = os.path.join(STATIC_DIR, static_file_name)
        uploaded_df = pd.read_csv(file_path).dropna(how="all")
        uploaded_df.columns = uploaded_df.columns.str.strip()
        st.success("📁 Static File Loaded")
        st.markdown("### 🧾 Static CSV Preview")
        st.dataframe(uploaded_df)
        file_to_upload = open(file_path, "rb")

# Aggregation mode selector
aggregation_mode = st.selectbox(
    "📊 Aggregation Source",
    options=["table"],
    format_func=lambda x: x.replace("_", " ").title()
)

# Trigger upload and analysis
if file_to_upload and st.button("🚀 Upload and Analyze"):
    response = requests.post(
        UPLOAD_ENDPOINT,
        params={"aggregation_mode": aggregation_mode},
        files={"file": (getattr(file_to_upload, 'name', 'file.csv'), file_to_upload, "text/csv")}
    )

    if response.ok:
        json_response = response.json()

        if "abnormal_records" in json_response:
            df_abnormal = pd.DataFrame(json_response["abnormal_records"])
            if not df_abnormal.empty:
                st.markdown("### ⚠️ Abnormal Records Detected")
                st.dataframe(df_abnormal)

                csv = df_abnormal.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Abnormal Records", csv, "abnormal_records.csv", "text/csv")

            if uploaded_df is not None:
                df_uploaded = uploaded_df.copy()
                df_uploaded["Transaction Date"] = df_uploaded["Transaction Date"].astype(str)

                total_records = len(df_uploaded)
                abnormal_records = len(df_abnormal)
                normal_records = total_records - abnormal_records

                df_overall = pd.DataFrame({
                    "Transaction Type": ["Abnormal", "Normal"],
                    "Count": [abnormal_records, normal_records]
                })

                st.markdown("### 📊 Overall Abnormal vs Normal Transactions")
                st.bar_chart(df_overall.set_index("Transaction Type"))

        else:
            st.success("✅ No abnormal transactions detected.")

    else:
        st.error(f"Upload failed: {response.text}")

# Close static file handle if used
if file_to_upload and not isinstance(file_to_upload, st.runtime.uploaded_file_manager.UploadedFile):
    file_to_upload.close()

st.markdown("---")
st.title("A Single Transaction Anomaly Detection Data Entry")

selected_date = st.date_input("📅 Select Transaction Date", value=date.today())
transaction_date_str = selected_date.strftime("%Y%m%d")
current_time_str = datetime.now().strftime("%H%M%S")
transaction_amount = st.number_input("💸 Enter Transaction Amount", min_value=0, value=100, step=100)
merchant_id = st.text_input("Enter Merchant ID", value="F000000018")
sub_merchant_id = st.text_input("Enter Sub_Merchant_ID", value="00990981771")

# Mode
mode = st.selectbox(
    "🧠 Choose Anomaly Detection Method",
    options=["rule", "rf"],
    index=0,
    format_func=lambda x: "Rule-Based" if x == "rule" else "Random Forest",
)

def insert_data(data, mode, aggregation_mode):
    try:
        response = requests.post(
            API_ENDPOINT,
            params={"mode": mode, "aggregation_mode": aggregation_mode},
            json=data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None

# Construct default payload
default_data = {
    "Transaction_Date": transaction_date_str,
    "Transaction_Time": current_time_str,
    "Transaction_Amount": transaction_amount,
    "Customer_Date_of_Birth": "90522",
    "Customer_Gender": 3,
    "Customer_Age": 15,
    "Customer_Region_City_County_District": "41463",
    "Merchant_Region_City_County_District": "41463",
    "Merchant_ID": merchant_id,
    "Sub_Merchant_ID": sub_merchant_id,
    "Transaction_Processing_Agency_VAN": 55,
    "Distance": 1
}

json_input = st.text_area(
    "✏️ Edit Transaction JSON:",
    value=json.dumps(default_data, indent=4),
    height=300
)

if st.button("Insert Data"):
    try:
        new_trans = json.loads(json_input)
        result = insert_data(new_trans, mode, aggregation_mode)
        if result:
            payload = result["payload"]
            st.success("✅ Transaction inserted successfully!")
            st.markdown(f"""
                <p><strong>🗓 Daily Threshold:</strong> {payload['daily_threshold_cache']:,} ₩</p>
                <p><strong>📅 Monthly Threshold:</strong> {payload['monthly_threshold_cache']:,} ₩</p>
            """, unsafe_allow_html=True)

            if payload['daily_exceeded']:
                st.warning("⚠️ Daily threshold exceeded!")
            if payload['monthly_exceeded']:
                st.warning("⚠️ Monthly threshold exceeded!")

            if payload["is_anomaly"]:
                st.error("🚨 Anomaly Detected!")
            else:
                st.success("✅ No anomaly detected.")

            if payload["triggered_rules"]:
                st.markdown("### 🧠 Triggered Rules")
                for rule in payload["triggered_rules"]:
                    st.markdown(f"- ❗ **{rule}**")

            with st.expander("Show raw response JSON"):
                st.json(result)
    except Exception as e:
        st.error(f"Error: {e}")
