#!/usr/bin/env python3

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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
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

# -------------------------------------------
# Jalgaon-Specific Extraction Utility Functions
# -------------------------------------------
EXPECTED_NCOLS = 8  # Suppose we want 8 consistent columns: indices 0..7

def fix_columns_for_page(df: pd.DataFrame, page_num: int) -> pd.DataFrame:
    """
    Force 'df' to have EXACTLY EXPECTED_NCOLS columns in the right positions.
    You can customize this logic depending on your PDF's pattern and page number.
    """
    current_ncols = df.shape[1]
    
    if current_ncols == EXPECTED_NCOLS:
        # Perfect, rename columns to 0..7
        df.columns = range(EXPECTED_NCOLS)
        return df

    elif current_ncols < EXPECTED_NCOLS:
        # Example logic: if it's missing one column, insert a blank
        df.columns = range(current_ncols)
        if current_ncols == 7:
            # Insert a blank column at index 5 (example)
            df.insert(5, "blank", "")
        df.columns = range(EXPECTED_NCOLS)
        return df

    else:
        # current_ncols > EXPECTED_NCOLS
        df.columns = range(current_ncols)
        if current_ncols == 9:
            # Maybe drop column #5
            df.drop(columns=[5], inplace=True)
        df.columns = range(EXPECTED_NCOLS)
        return df

def is_date(val: str) -> bool:
    """Simple check if val is in DD/MM/YYYY or DD-MM-YYYY."""
    val = val.strip()
    return bool(re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", val))

def merge_multiline_rows(df: pd.DataFrame, date_col: int = 0, partic_col: int = 2) -> pd.DataFrame:
    """Merge rows whose first cell is not a date into the previous row's 'particular' text."""
    merged_rows = []
    for i in range(len(df)):
        row = df.iloc[i].copy()
        first_cell = str(row[date_col]).strip()
        if not is_date(first_cell):
            if merged_rows:
                merged_rows[-1][partic_col] = (
                    str(merged_rows[-1][partic_col]) + " " + str(row[partic_col])
                )
            else:
                merged_rows.append(row)
        else:
            merged_rows.append(row)
    return pd.DataFrame(merged_rows, columns=df.columns)

def is_valid_transaction(row) -> bool:
    """
    Decide if a row is a 'valid transaction':
      - Column 0 is a valid date
      - Column 3 is one of ["T", "L", "C"]
    Adjust these rules as needed.
    """
    # Check date in column 0
    if not is_date(str(row[0])):
        return False
    # Check transaction code in column 3
    valid_codes = {"T", "L", "C"}
    if str(row[3]) not in valid_codes:
        return False
    return True

def filter_valid_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply is_valid_transaction to each row; keep only rows that pass."""
    mask = df.apply(is_valid_transaction, axis=1)
    return df[mask].copy()

# -------------------------------------------
# Other Extraction and Cleaning Functions
# -------------------------------------------

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

def extract_table(pdf_file, flavor="stream", pages="all", **kwargs):
    tables = camelot.read_pdf(pdf_file, flavor=flavor, pages=pages, **kwargs)
    if len(tables) > 0:
        return pd.concat([table.df for table in tables], ignore_index=True)
    return None

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

# -------------------------------------------
# Utility Functions for Production Setup
# -------------------------------------------
def normalize_identifier(text):
    """Normalize text (email, company) to be used in table names."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text

# -------------------------------------------
# FastAPI Endpoints
# -------------------------------------------

# Endpoint to serve the HTML page for file upload and display of extracted data.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Table Extraction Test</title>
    </head>
    <body>
        <h1>Upload Your PDF</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <label>Email: <input type="text" name="email" placeholder="Enter your email" required></label><br>
            <label>Company: <input type="text" name="company" placeholder="Enter company name" required></label><br>
            <!-- Hidden inputs for other form values -->
            <input type="hidden" name="uploaded_file" value="uploaded.pdf">
            <input type="hidden" name="user_group" value="gold">
            <label>File: <input type="file" name="file" accept="application/pdf" required></label><br>
            <button type="submit">Submit</button>
        </form>
        <hr>
        <h2>Extracted Data</h2>
        <div id="result"></div>
        <script>
            document.getElementById("uploadForm").addEventListener("submit", async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                const response = await fetch("/process-pdf", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();
                document.getElementById("result").innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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

# Main endpoint: processes uploaded PDF, saves extracted data to files,
# and returns the parsed output as JSON.
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
        
        # If unsupported bank type, return a message
        if bank_type == "unknown":
            return JSONResponse(status_code=200, content={"status": "unsupported bank type"})
        
        df = None
        
        if bank_type == "jalgaon":
            # ------ Jalgaon extraction using the new logic ------
            tables = camelot.read_pdf(pdf_file_path, flavor="stream", pages="all")
            if not tables or len(tables) == 0:
                print("No tables found with 'stream' flavor. Trying 'lattice'.")
                tables = camelot.read_pdf(pdf_file_path, flavor="lattice", pages="all")
                if not tables or len(tables) == 0:
                    raise HTTPException(status_code=400, detail="No tables extracted from PDF.")
            
            page_tables = {}
            for table in tables:
                pnum = table.page
                if pnum not in page_tables:
                    df_page = table.df.copy()
                    # Fix columns to expected number
                    df_page = fix_columns_for_page(df_page, pnum)
                    # Add page column for traceability
                    df_page["page"] = pnum
                    page_tables[pnum] = df_page
            
            if not page_tables:
                raise HTTPException(status_code=400, detail="No valid tables found on any page.")
            
            # Combine all pages into one DataFrame
            combined_df = pd.concat(list(page_tables.values()), ignore_index=True)
            # Merge multi-line rows (assuming date is in column 0 and particulars in column 2)
            combined_df = merge_multiline_rows(combined_df, date_col=0, partic_col=2)
            # Drop duplicate rows if any
            combined_df.drop_duplicates(inplace=True)
            # Filter for valid transactions using the new filter logic
            filtered_df = filter_valid_transactions(combined_df)
            df = filtered_df
            # ----------------------------------------------------
        
        elif bank_type == "axis bank":
            df = extract_table(pdf_file_path, flavor="stream", pages="all")
            if df is None or df.empty:
                df = extract_table(pdf_file_path, flavor="lattice", pages="all")
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
            # Use existing cleaning for axis bank
            df = merge_axis_rows(df, date_col="date")
        
        elif bank_type in ["idbi", "sbi new", "sbi", "pnb", "union bank"]:
            df = extract_tables_pdfplumber(pdf_file_path)
            if bank_type in ["sbi new", "sbi"]:
                df = transform_sbi(df)
        
        elif bank_type == "icici3":
            df = extract_icici3_with_pdfplumber(pdf_file_path)
            if df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
        
        else:
            df = extract_table(pdf_file_path, flavor="stream", pages="all")
            if df is None or df.empty:
                df = extract_table(pdf_file_path, flavor="lattice", pages="all")
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data extracted from PDF.")
        
        # Convert DataFrame to JSON (list of records) for final output
        parsed_data = df.to_dict(orient="records")
        
        # Save final processed data to a JSON file
        with open("extracted_data.json", "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        
        return JSONResponse(status_code=200, content={"status": "success", "parsed_data": parsed_data})
    
    finally:
        os.remove(pdf_file_path)
