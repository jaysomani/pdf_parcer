import os
import re
import json
import base64
import tempfile
import pdfplumber
import camelot
import pandas as pd
import numpy as np
from io import BytesIO
import psycopg2
from psycopg2.extras import execute_values
import boto3

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()

# Add CORS middleware here:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = Mangum(app)

# ---------------------------
# PDF Table Extraction Functions
# ---------------------------
def deduplicate_columns(df):
    new_cols = []
    counts = {}
    for col in df.columns:
        if col in counts:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

def remove_newlines(df):
    return df.applymap(lambda x: str(x).replace("\n", " ") if isinstance(x, str) else x)

def extract_tables_pdfplumber(pdf_file):
    all_dfs = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if len(table) < 2:
                        print(f"Page {page_num}: Skipping table (not enough rows).")
                        continue
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df = deduplicate_columns(df)
                    df = remove_newlines(df)
                    df["page"] = page_num
                    all_dfs.append(df)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    return combined_df

# ---------------------------
# Bank Detection & Extraction Functions
# ---------------------------
def detect_bank_type(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text() or ""
    text_lower = text.lower()
    bank_patterns = {
        "date narration chq/ref no balance": "kotak",
        "date mode** particulars deposits withdrawals balance": "icici3",
        "date transaction reference ref.no./chq.no. credit debit balance": "sbi new",
        "date narration chq./ref.no. valuedt withdrawalamt. depositamt. closingbalance": "hdfc",
        "serial transaction value description cheque debit credit balance": "bob",
        "trndate valuedt particular insno / type withdrawals deposit balance": "jalgaon",
        "txn no. txn date description branch name balance": "pnb",
        "srl txn date value date description cr/dr amount (inr) balance (inr)": "idbi",
        "tran date chq no particulars debit credit balance init.": "axis bank",
        "txn date value date description ref no./cheque branch debit credit balance": "sbi"
    }
    for pattern, bank in bank_patterns.items():
        if pattern in text_lower:
            return bank
    if ("transaction id" in text_lower and "txn posted date" in text_lower
        and "chequeno." in text_lower and "transaction amount(inr)" in text_lower
        and "available balance(inr)" in text_lower):
        return "icici3"
    return "unknown"

def extract_icici3_with_pdfplumber(pdf_file):
    all_rows = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables:
                all_rows.extend(tbl)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
    return df

def extract_table(pdf_file, flavor="stream", pages="1", **kwargs):
    tables = camelot.read_pdf(pdf_file, flavor=flavor, pages=pages, **kwargs)
    if len(tables) > 0:
        return pd.concat([table.df for table in tables], ignore_index=True)
    return None

# ---------------------------
# Utility Functions for Data Cleaning
# ---------------------------
def find_header_index(df, header_keywords):
    for i, row in df.iterrows():
        row_str = " ".join(str(x).lower() for x in row if pd.notna(x))
        matches = sum(1 for kw in header_keywords if kw in row_str)
        if matches >= len(header_keywords) - 1:
            return i
    return None

def merge_multi_line_rows(transactions, desc_col_name):
    def is_date(val):
        val = str(val).strip()
        return bool(re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", val))
    merged_rows = []
    i = 0
    while i < len(transactions):
        row = transactions.iloc[i].copy()
        first_cell = row.iloc[0]
        if is_date(first_cell):
            j = i + 1
            while j < len(transactions) and not is_date(transactions.iloc[j, 0]):
                continuation_desc = str(transactions.iloc[j].get(desc_col_name, "")).strip()
                if continuation_desc:
                    row[desc_col_name] = str(row.get(desc_col_name, "")) + " " + continuation_desc
                j += 1
            merged_rows.append(row)
            i = j
        else:
            i += 1
    merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    return merged_df

def clean_extracted_table(df, bank_type):
    if bank_type == "jalgaon":
        header_keywords = ["trndate", "valuedt", "withdrawals", "deposit", "balance"]
        desc_col = "particular"
    elif bank_type == "kotak":
        header_keywords = ["date", "narration", "chq/ref no", "balance"]
        desc_col = "narration"
    else:
        header_keywords = ["date", "description", "balance"]
        desc_col = "description"
    header_index = find_header_index(df, header_keywords)
    if header_index is None:
        print("No matching header row found; returning raw extracted table.")
        return df
    transactions = df.iloc[header_index+1:].copy()
    header_row = df.iloc[header_index].tolist()
    transactions.columns = header_row
    transactions.columns = [str(col).strip().lower() for col in transactions.columns]
    if desc_col not in transactions.columns:
        desc_col = transactions.columns[1]
    merged_df = merge_multi_line_rows(transactions, desc_col)
    return merged_df

def merge_axis_rows(df, date_col="date"):
    def is_valid_date(val):
        return bool(re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", str(val).strip()))
    merged_rows = []
    i = 0
    while i < len(df):
        row = df.iloc[i].copy()
        if is_valid_date(row[date_col]):
            j = i + 1
            while j < len(df) and not is_valid_date(df.iloc[j][date_col]):
                for col in df.columns:
                    if col != date_col:
                        row[col] = str(row[col]) + " " + str(df.iloc[j][col])
                j += 1
            merged_rows.append(row)
            i = j
        else:
            i += 1
    merged_df = pd.DataFrame(merged_rows, columns=df.columns)
    return merged_df

def transform_sbi(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ['txn date', 'description', 'debit', 'credit', 'balance']
    for col in required:
        if col not in df.columns:
            print(f"Column {col} not found in the DataFrame.")
            return df
    for col in ['debit', 'credit', 'balance']:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=True)
            .str.strip()
            .replace("", np.nan)
            .astype(float, errors="ignore")
        )
    new_rows = []
    for _, row in df.iterrows():
        txn_date = row.get('txn date', "").strip()
        description = row.get('description', "").strip()
        debit = row.get('debit', 0) or 0
        credit = row.get('credit', 0) or 0
        if not txn_date:
            continue
        if credit > 0:
            tx_type = "receipt"
            amt = credit
        elif debit > 0:
            tx_type = "payment"
            amt = debit
        else:
            continue
        new_rows.append({
            "txn_date": txn_date,
            "description": description,
            "type": tx_type,
            "amount": amt,
            "ledger": ""
        })
    return pd.DataFrame(new_rows, columns=["txn_date", "description", "type", "amount", "ledger"])

def transform_jalgaon(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ['trndate', 'particular', 'deposit', 'withdrawals']
    for col in required:
        if col not in df.columns:
            print(f"Column '{col}' not found in the DataFrame.")
            return df
    for col in ['deposit', 'withdrawals']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=True).str.strip().replace("", np.nan),
            errors="coerce"
        )
    new_rows = []
    for _, row in df.iterrows():
        txn_date = str(row.get('trndate', "")).strip()
        description = str(row.get('particular', "")).strip()
        deposit_val = row.get('deposit', 0) or 0
        withdrawal_val = row.get('withdrawals', 0) or 0
        if not txn_date:
            continue
        if deposit_val > 0:
            tx_type = "receipt"
            amt = deposit_val
        elif withdrawal_val > 0:
            tx_type = "payment"
            amt = withdrawal_val
        else:
            continue
        new_rows.append({
            "txn_date": txn_date,
            "description": description,
            "type": tx_type,
            "amount": amt,
            "ledger": ""
        })
    return pd.DataFrame(new_rows, columns=["txn_date", "description", "type", "amount", "ledger"])

# ---------------------------
# Utility Functions for Production Setup
# ---------------------------
def normalize_identifier(text):
    """Normalize text (email, company) to be used in table names."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text

# ---------------------------
# FastAPI Endpoints
# ---------------------------

# Dummy endpoint to test if API is running.
@app.get("/hello")
def hello():
    return {"message": "Hello, world! API is working."}

# Dummy endpoint to echo back the uploaded file details.
@app.post("/echo-file")
async def echo_file(file: UploadFile = File(...)):
    file_content = await file.read()
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    return JSONResponse(status_code=200, content={
        "filename": file.filename,
        "content_base64": encoded_content
    })

# Main endpoint: processes uploaded PDF and returns the parsed output as JSON.
@app.post("/process-pdf")
async def process_pdf(
    email: str = Form(...),
    company: str = Form(...),
    uploaded_file: str = Form("uploaded.pdf"),
    user_group: str = Form("gold"),
    file: UploadFile = File(...)
):
    # Read uploaded PDF data
    pdf_data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_file_path = tmp.name
        tmp.write(pdf_data)
    
    try:
        bank_type = detect_bank_type(pdf_file_path)
        print("Detected bank type:", bank_type)
        
        # If unsupported bank type, return a message (optionally, upload to S3 if needed)
        if bank_type == "unknown":
            return JSONResponse(status_code=200, content={"status": "unsupported bank type"})
        
        # Process PDF based on bank type
        df = None
        if bank_type == "axis bank":
            df = extract_table(pdf_file_path, flavor="stream", pages="1")
            if df is None or df.empty:
                df = extract_table(pdf_file_path, flavor="lattice", pages="1")
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
            df = clean_extracted_table(df, bank_type)
            df = merge_axis_rows(df, date_col="date")
        elif bank_type in ["idbi", "sbi new", "sbi", "pnb", "union bank"]:
            df = extract_tables_pdfplumber(pdf_file_path)
            if bank_type in ["sbi new", "sbi"]:
                df = transform_sbi(df)
        elif bank_type == "jalgaon":
            df = extract_table(pdf_file_path, flavor="stream", pages="1")
            if df is None or df.empty:
                df = extract_table(pdf_file_path, flavor="lattice", pages="1")
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
            df = clean_extracted_table(df, bank_type)
            df = transform_jalgaon(df)
        elif bank_type == "icici3":
            df = extract_icici3_with_pdfplumber(pdf_file_path)
            if df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
        else:
            df = extract_table(pdf_file_path, flavor="stream", pages="1")
            if df is None or df.empty:
                df = extract_table(pdf_file_path, flavor="lattice", pages="1")
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
            df = clean_extracted_table(df, bank_type)
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data extracted from PDF.")
        
        # Convert DataFrame to JSON (list of records)
        parsed_data = df.to_dict(orient="records")
        return JSONResponse(status_code=200, content={"status": "success", "parsed_data": parsed_data})
    
    finally:
        os.remove(pdf_file_path)
