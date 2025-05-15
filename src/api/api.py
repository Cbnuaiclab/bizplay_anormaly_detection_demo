from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from collections import defaultdict
import pandas as pd
import joblib
import numpy as np
from decimal import Decimal
import csv, io
import time
from fastapi.responses import JSONResponse
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz

from fastapi import Query
from typing import List
import re

scheduler = BackgroundScheduler(timezone="Asia/Seoul")  # adjust timezone if needed

def clear_daily_cache():
    daily_threshold_cache.clear()
    print("✅ Cleared daily cache at midnight")

def clear_monthly_cache():
    monthly_threshold_cache.clear()
    print("✅ Cleared monthly cache on 1st day of month")

DB_USER = 'postgres'
DB_PASSWORD = 'postgres123'
DB_HOST = '10.255.78.58'
DB_PORT = '9002'
DB_NAME = 'anomaly_detection'

connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

daily_threshold_cache = defaultdict(Decimal)
monthly_threshold_cache = defaultdict(Decimal)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    with engine.begin() as conn:
        conn.execute(text("""
            TRUNCATE TABLE monthly_aggregation;

            INSERT INTO monthly_aggregation (month, sub_merchant_id, total_amount , transaction_count, last_updated)
            SELECT
                TO_CHAR(TO_DATE("Transaction Date", 'YYYYMMDD'), 'YYYYMM') AS month,
                COALESCE("Sub-Merchant ID", "Merchant ID") AS sub_merchant_id,
                SUM("Transaction Amount") AS "Total Amount",
                COUNT(*) AS transaction_count,
                CURRENT_TIMESTAMP
            FROM "BizPlay_2025_JAN"
            GROUP BY month, COALESCE("Sub-Merchant ID", "Merchant ID");
            """))
         
        conn.execute(
            text(
                """
                    TRUNCATE TABLE daily_aggregation;

                    INSERT INTO daily_aggregation (transaction_date, sub_merchant_id, total_amount, transaction_count, last_updated)
                    SELECT
                        "Transaction Date" AS transaction_date,
                        COALESCE("Sub-Merchant ID", "Merchant ID") AS sub_merchant_id,
                        SUM("Transaction Amount"),
                        COUNT(*),
                        CURRENT_TIMESTAMP
                    FROM "BizPlay_2025_JAN"
                    GROUP BY transaction_date, COALESCE("Sub-Merchant ID", "Merchant ID");
                """
            )
        )

        # Schedule daily reset at 00:00
        scheduler.add_job(clear_daily_cache, trigger="cron", hour=0, minute=0)

        # Schedule monthly reset on the 1st at 00:01
        scheduler.add_job(clear_monthly_cache, trigger="cron", day=1, hour=0, minute=0)

        scheduler.start()
 
class TransactionData(BaseModel):
    Transaction_Date: str
    Transaction_Time: str
    Transaction_Amount: Decimal
    Customer_Date_of_Birth: str
    Customer_Gender: int
    Customer_Age: int
    Customer_Region_City_County_District: str
    Merchant_Region_City_County_District: str
    Merchant_ID: str
    Sub_Merchant_ID: str
    Transaction_Processing_Agency_VAN: int
    Distance: int

# Get monthly aggregation from Postgres
def validate_monthly_threshold(sub_merchant_id: str, transaction_date: str, aggregation_mode: str = "table"):
    table_name = {
        "table": "monthly_aggregation",
        "view": "monthly_summary_view",
        "materialized_view": "monthly_summary"
    }[aggregation_mode]

    query = text(f"""
        SELECT total_amount 
        FROM {table_name}
        WHERE sub_merchant_id = :sub_merchant_id 
        AND "month" = :month
    """)
 
    with engine.begin() as conn:
        result = conn.execute(query, {
            "sub_merchant_id": sub_merchant_id,
            "month": transaction_date[:6]
        }).fetchone()
    return result[0] if result != None else 0


def validate_daily_threshold(sub_merchant_id: str, transaction_date: str, aggregation_mode: str = "table"):
    table_name = {
        "table": "daily_aggregation",
        "view": "daily_summary_view",
        "materialized_view": "daily_summary"
    }[aggregation_mode]

    query = text(f"""
        SELECT total_amount 
        FROM {table_name}
        WHERE sub_merchant_id = :sub_merchant_id 
        AND transaction_date = :transaction_date
    """)

    with engine.begin() as conn:
        result = conn.execute(query, {
            "sub_merchant_id": sub_merchant_id,
            "transaction_date": transaction_date
        }).fetchone()
    
    return result[0] if result != None else 0 


def insert_into_database(data: TransactionData, mode: str = "rule", aggregation_mode: str = "table"):
    try:
        triggered_rules = []
        trans_date = data.Transaction_Date
        # Use Sub_Merchant_ID if available, else fallback to Merchant_ID
        sid = data.Sub_Merchant_ID if data.Sub_Merchant_ID else data.Merchant_ID

        # Define threshold keys using fallback-safe sid
        key = (trans_date, sid)
        monthly_key = (trans_date[:6], sid)
        
        insert_query = text("""
            INSERT INTO "BizPlay_2025_JAN" (
                "Transaction Date", "Transaction Time", "Transaction Amount", 
                "Customer Date of Birth", "Customer Gender", "Customer Age", 
                "Customer Region (City/County/District)", "Merchant Region (City/County/District)", 
                "Merchant ID", "Sub-Merchant ID", "Transaction Processing Agency (VAN)", "Distance"
            ) VALUES (
                :transaction_date, :transaction_time, :transaction_amount,
                :customer_dob, :customer_gender, :customer_age,
                :customer_region, :merchant_region,
                :merchant_id, :sub_merchant_id, :van, :distance
            );
        """)

        params = {
            "transaction_date": data.Transaction_Date,
            "transaction_time": data.Transaction_Time,
            "transaction_amount": data.Transaction_Amount,
            "customer_dob": data.Customer_Date_of_Birth,
            "customer_gender": data.Customer_Gender,
            "customer_age": data.Customer_Age,
            "customer_region": data.Customer_Region_City_County_District,
            "merchant_region": data.Merchant_Region_City_County_District,
            "merchant_id": data.Merchant_ID,
            "sub_merchant_id": data.Sub_Merchant_ID,
            "van": data.Transaction_Processing_Agency_VAN,
            "distance": data.Distance
        }
        with engine.begin() as conn:
            conn.execute(insert_query, params)
        
        if key not in daily_threshold_cache:
            daily_amount = validate_daily_threshold(sid, trans_date, aggregation_mode)
            daily_threshold_cache[key] = daily_amount
        daily_threshold_cache[key] += data.Transaction_Amount
        
        if key not in monthly_threshold_cache:
            monthly_amount = validate_monthly_threshold(sid, trans_date, aggregation_mode)
            monthly_threshold_cache[monthly_key] = monthly_amount
        monthly_threshold_cache[monthly_key] += data.Transaction_Amount
        
        daily_exceeded = daily_threshold_cache[key] > 500000
        monthly_exceeded = monthly_threshold_cache[monthly_key] > 3000000

        print("daily_threshold_cache: ",daily_threshold_cache)
        print("monthly_threshold_cachely: ", monthly_threshold_cache)
        
        if daily_exceeded:
            triggered_rules.append("Dailylimit")
        elif monthly_exceeded:
            triggered_rules.append("MonthlyLimit")
        
        # Anomaly detection result
        is_anomaly = 0
        source = "none"
        rf_prediction = None
        rf_probability = None

        rules_violated = []
        if mode == "rule":
            rules_violated = rules_based(data, triggered_rules)
            if rules_violated:
                is_anomaly = 1
                source = "rule"
        else:  # mode == "rf"
            rf_result = random_forest_classification(data)
            rf_prediction = rf_result["rf_prediction"]
            rf_probability = rf_result["rf_probability"]
            if rf_prediction == 1:
                is_anomaly = 1
                source = "random_forest"
        return {
            "daily_threshold_cache":daily_threshold_cache[key],
            "monthly_threshold_cache": int(monthly_threshold_cache[monthly_key]),
            "daily_exceeded": bool(daily_exceeded),
            "monthly_exceeded": bool(monthly_exceeded),
            "triggered_rules": rules_violated if mode == 'rule' else None,
            "rf_prediction": rf_prediction,
            "rf_probability": rf_probability,
            "source": source,
            "is_anomaly": is_anomaly
        }
        
    except Exception as e:
        print("e: ",e)
        raise HTTPException(status_code=500, detail=str(e))

def random_forest_classification(data):
    model = joblib.load("../../models/random_forest_anomaly_model.pkl")
    scaler = joblib.load("../../models/scaler.pkl")

    tx_df = pd.DataFrame([{
        'Log_Transaction_Amount': np.log1p(float(abs(data.Transaction_Amount))),
        'Distance': data.Distance,
        'Customer Age': data.Customer_Age,
        'Transaction_Hour': int(data.Transaction_Time) // 10000,
        'Customer_Encoded': int(data.Customer_Region_City_County_District),
        'Merchant_Encoded': int(data.Merchant_Region_City_County_District),
        'Sub-Merchant ID': int(data.Sub_Merchant_ID),
    }])

    cols_to_scale = ['Log_Transaction_Amount', 'Distance', 'Customer Age', 'Transaction_Hour',
                     'Customer_Encoded', 'Merchant_Encoded', 'Sub-Merchant ID']
    tx_df[cols_to_scale] = scaler.transform(tx_df[cols_to_scale])

    prob = model.predict_proba(tx_df)[0][1]
    pred = model.predict(tx_df)[0]
    # print("prob: ",prob)

    return {
        "rf_prediction": int(pred),
        "rf_probability": float(prob)
    }


def rules_based(data, triggered_rules):
    age = data.Customer_Age
    time = int(data.Transaction_Time)
    hour = time // 10000
    region_cust = data.Customer_Region_City_County_District
    region_merchant = data.Merchant_Region_City_County_District

    # --- A. Age + Time ---
    if age < 8 and hour >= 22: 
        triggered_rules.append("AgeTime")

    # --- B. Restricted Hours ---
    if hour >= 23 or hour < 6:
        triggered_rules.append("RestrictedHour")
        
    # --- C. Lunch by Financial Aid ---
    if 11 <= hour < 13:
        triggered_rules.append("LunchTimeAid")

    # --- D. Region Mismatch (excluding online merchants)
    if region_cust != region_merchant and region_merchant != "ONLINE":
        triggered_rules.append("RegionMismatch")
    return triggered_rules



@app.post("/transactions/")
async def create_transaction(data: TransactionData, mode: str = "rule", aggregation_mode: str = "table"):
    json_data = insert_into_database(data, mode, aggregation_mode)
    return {"message": "Transaction created successfully","payload":json_data}

from fastapi import UploadFile, File
import csv
import io

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), aggregation_mode: str = "table"):
    try:
        required_columns = {
            "Transaction Date",
            "Transaction Time",
            "Transaction Amount",
            "Customer Date of Birth",
            "Customer Gender",
            "Customer Age",
            "Customer Region (City/County/District)",
            "Merchant Region (City/County/District)",
            "Merchant ID",
            "Sub-Merchant ID",
            "Transaction Processing Agency (VAN)",
            "Distance"
        }

        # Read and parse CSV content
        content = await file.read()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

        # Strip whitespace and normalize headers
        reader.fieldnames = [header.strip() for header in reader.fieldnames]
        uploaded_columns = set(reader.fieldnames)

        # Validate column headers
        missing_columns = required_columns - uploaded_columns
        if missing_columns:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"The uploaded file is missing required columns: {', '.join(missing_columns)}"
                }
            )


        total_inserted = 0
        daily_violations = 0
        monthly_violations = 0

        start_time = time.perf_counter()
        abnormal_records = []

        for row in reader:
            if all(value.strip() == "" for value in row.values()):
                continue  # Skip blank row
            try:
                print("row: ",row)
                data = TransactionData(
                    Transaction_Date=row["Transaction Date"],
                    Transaction_Time=row["Transaction Time"],
                    Transaction_Amount=Decimal(row["Transaction Amount"]),
                    Customer_Date_of_Birth=row["Customer Date of Birth"],
                    Customer_Gender=int(row["Customer Gender"]),
                    Customer_Age=int(row["Customer Age"]),
                    Customer_Region_City_County_District=row["Customer Region (City/County/District)"],
                    Merchant_Region_City_County_District=row["Merchant Region (City/County/District)"],
                    Merchant_ID=row["Merchant ID"],
                    Sub_Merchant_ID=row["Sub-Merchant ID"],
                    Transaction_Processing_Agency_VAN=int(row["Transaction Processing Agency (VAN)"]),
                    Distance=int(row["Distance"]),
                )
                result = insert_into_database(data, mode="rule", aggregation_mode=aggregation_mode)
                if result["daily_exceeded"] or result["monthly_exceeded"] or result["is_anomaly"]:
                    row["Daily Exceeded"] = result["daily_exceeded"]
                    row["Monthly Exceeded"] = result["monthly_exceeded"]
                    row["Anomaly"] = result["is_anomaly"]
                    row["Source"] = result["source"]
                    row["Triggered Rules"] = ", ".join(result["triggered_rules"]) if result["triggered_rules"] else ""
                    abnormal_records.append(row)

                if result["daily_exceeded"]:
                    daily_violations += 1
                if result["monthly_exceeded"]:
                    monthly_violations += 1

                total_inserted += 1

                triggered_rules=[]
                # rules_violated = rules_based(data, triggered_rules)
                # rf_result = random_forest_classification(data)
            except Exception as e:
                print(f"Skipping row: {e}")
                continue

        end_time = time.perf_counter()

        # print("daily_threshold_cache",daily_threshold_cache)
        # print("monthly_threshold_cache",monthly_threshold_cache)

        normal_count = total_inserted - len(abnormal_records)

        # Return 400 if no rows were inserted
        if total_inserted == 0:
            return JSONResponse(
                status_code=HTTP_400_BAD_REQUEST,
                content={"error": "No valid transaction rows found in the CSV file."}
            )

        # Return 200 OK with summary
        return JSONResponse(
            status_code=HTTP_200_OK,
            content={
                "message": "CSV processed successfully.",
                "total_inserted": total_inserted,
                "daily_threshold_exceeded": daily_violations,
                "monthly_threshold_exceeded": monthly_violations,
                "total_execution_time_ms": round((end_time - start_time) * 1000, 2),
                "abnormal_records": abnormal_records
            }
        )


    except Exception as e:
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.get("/stats/")
def get_system_stats():
    return {
        "current_daily_cache_keys": len(daily_threshold_cache),
        "current_monthly_cache_keys": len(monthly_threshold_cache),
        "cache_memory_mb": round(
            (daily_threshold_cache.__sizeof__() + monthly_threshold_cache.__sizeof__()) / 1024 / 1024, 2
        )
    }

@app.post("/admin/reset-system/")
def reset_system():
    try:
        # Clear in-memory caches
        daily_threshold_cache.clear()
        monthly_threshold_cache.clear()

        # Truncate aggregation tables
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE daily_aggregation"))
            conn.execute(text("""
                    INSERT INTO daily_aggregation (transaction_date, sub_merchant_id, total_amount, transaction_count, last_updated)
                    SELECT
                        "Transaction Date" AS transaction_date,
                        COALESCE("Sub-Merchant ID", "Merchant ID") AS sub_merchant_id,
                        SUM("Transaction Amount"),
                        COUNT(*),
                        CURRENT_TIMESTAMP
                    FROM "BizPlay_2025_JAN"
                    GROUP BY transaction_date, COALESCE("Sub-Merchant ID", "Merchant ID");
                """))
            conn.execute(text("TRUNCATE TABLE monthly_aggregation"))
            conn.execute(text("""
                INSERT INTO monthly_aggregation (month, sub_merchant_id, total_amount , transaction_count, last_updated)
                SELECT
                    TO_CHAR(TO_DATE("Transaction Date", 'YYYYMMDD'), 'YYYYMM') AS month,
                    COALESCE("Sub-Merchant ID", "Merchant ID") AS sub_merchant_id,
                    SUM("Transaction Amount") AS "Total Amount",
                    COUNT(*) AS transaction_count,
                    CURRENT_TIMESTAMP
                FROM "BizPlay_2025_JAN"
                GROUP BY month, COALESCE("Sub-Merchant ID", "Merchant ID");
            """))

        return JSONResponse(
            status_code=200,
            content={"message": "✅ System reset: caches cleared and aggregation tables truncated."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"❌ Failed to reset system: {str(e)}"}
        )