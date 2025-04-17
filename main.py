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
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = Mangum(app)

# -------------------------------------------
# Helper Functions
# -------------------------------------------

def add_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    def get_type(row):
        try:
            deposit = float(str(row["deposit"]).replace(",", "").strip() or 0)
        except:
            deposit = 0
        try:
            withdrawal = float(str(row["withdrawal"]).replace(",", "").strip() or 0)
        except:
            withdrawal = 0
        if deposit > 0:
            return "receipt"
        elif withdrawal > 0:
            return "payment"
        else:
            return "unknown"
    df["type"] = df.apply(get_type, axis=1)
    return df

def add_amount_column(df: pd.DataFrame) -> pd.DataFrame:
    def get_amount(row):
        if row["type"] == "receipt":
            return row["deposit"]
        elif row["type"] == "payment":
            return row["withdrawal"]
        else:
            return None
    df["amount"] = df.apply(get_amount, axis=1)
    return df[["date", "description", "balance", "type", "amount"]]

# -------------------------------------------
# Jalgaon‑Specific Extraction Utility Functions
# -------------------------------------------
EXPECTED_NCOLS = 8  # Expect 8 columns

def fix_columns_for_page(df: pd.DataFrame, page_num: int) -> pd.DataFrame:
    current_ncols = df.shape[1]
    print(f"[DEBUG] Page {page_num} - Original shape: {df.shape}")
    print(f"[DEBUG] Page {page_num} - Head before fix:\n{df.head(1)}")
    if current_ncols == EXPECTED_NCOLS:
        df.columns = list(range(EXPECTED_NCOLS))
    elif current_ncols < EXPECTED_NCOLS:
        for i in range(current_ncols, EXPECTED_NCOLS):
            df[i] = ""
        df = df[list(range(EXPECTED_NCOLS))]
        df.columns = list(range(EXPECTED_NCOLS))
        print(f"[DEBUG] Page {page_num} - After padding: {df.shape}")
    else:
        if current_ncols == EXPECTED_NCOLS + 1:
            df = df.drop(columns=[5])
        else:
            df = df.iloc[:, :EXPECTED_NCOLS]
        df.columns = list(range(EXPECTED_NCOLS))
        print(f"[DEBUG] Page {page_num} - After trimming: {df.shape}")
    return df

def is_date(val: str) -> bool:
    return bool(re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", val.strip()))

def merge_multiline_rows(df: pd.DataFrame, date_col: int = 0, partic_col: int = 2) -> pd.DataFrame:
    merged = []
    for i in range(len(df)):
        row = df.iloc[i].copy()
        if not is_date(str(row[date_col]).strip()):
            if merged:
                merged[-1][partic_col] = f"{merged[-1][partic_col]} {row[partic_col]}"
            else:
                merged.append(row)
        else:
            merged.append(row)
    return pd.DataFrame(merged, columns=df.columns)

def is_valid_transaction(row) -> bool:
    if not is_date(str(row[0]).strip()):
        return False
    if str(row[3]) not in {"T", "L", "C"}:
        return False
    return True

def filter_valid_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.apply(is_valid_transaction, axis=1)].copy()

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
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

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
    return pd.DataFrame(all_rows[1:], columns=all_rows[0])

def extract_table(pdf_file, flavor="stream", pages="all", **kwargs):
    tables = camelot.read_pdf(pdf_file, flavor=flavor, pages=pages, **kwargs)
    if len(tables) > 0:
        return pd.concat([t.df for t in tables], ignore_index=True)
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
                        row[col] = f"{row[col]} {df.iloc[j][col]}"
                j += 1
            merged_rows.append(row)
            i = j
        else:
            i += 1
    return pd.DataFrame(merged_rows, columns=df.columns)

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

def normalize_identifier(text):
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", text)

# -------------------------------------------
# FastAPI Endpoints
# -------------------------------------------

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

@app.get("/hello")
def hello():
    return {"message": "Hello, world! API is working."}

@app.post("/echo-file")
async def echo_file(file: UploadFile = File(...)):
    file_content = await file.read()
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    return JSONResponse(status_code=200, content={
        "filename": file.filename,
        "content_base64": encoded_content
    })

@app.post("/process-pdf")
async def process_pdf(
    email: str = Form(...),
    company: str = Form(...),
    uploaded_file: str = Form("uploaded.pdf"),
    user_group: str = Form("gold"),
    file: UploadFile = File(...)
):
    pdf_data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_file_path = tmp.name
        tmp.write(pdf_data)

    try:
        bank_type = detect_bank_type(pdf_file_path)
        print(f"[DEBUG] Detected bank type: {bank_type!r}")

        if bank_type == "unknown":
            return JSONResponse(status_code=200, content={"status": "unsupported bank type"})

        df = None

        if bank_type == "jalgaon":
            # ——— Jalgaon extraction with robust lattice + merge ——— #
            tables = camelot.read_pdf(
                pdf_file_path,
                flavor="lattice",
                pages="all",
                dpi=400,
                line_scale=50,
                edge_tol=200,
                split_text=False
            )
            print(f"[DEBUG] Lattice found {len(tables)} tables")

            if not tables:
                print("[DEBUG] Lattice failed—falling back to stream")
                tables = camelot.read_pdf(
                    pdf_file_path,
                    flavor="stream",
                    pages="all",
                    strip_text="\n"
                )
                print(f"[DEBUG] Stream found {len(tables)} tables")

            # Optional: dump images for visual debug
            # for i, tbl in enumerate(tables):
            #     tbl.to_image(f"/tmp/page{tbl.page}_tbl{i}.png").save()

            page_chunks = defaultdict(list)
            for tbl in tables:
                df_chunk = tbl.df.copy()
                df_fixed = fix_columns_for_page(df_chunk, tbl.page)
                df_fixed["page"] = tbl.page
                page_chunks[tbl.page].append(df_fixed)

            if not page_chunks:
                raise HTTPException(status_code=400, detail="No valid tables extracted")

            combined = pd.concat(
                [pd.concat(chunks, ignore_index=True) for chunks in page_chunks.values()],
                ignore_index=True
            )
            print(f"[DEBUG] Combined shape before merge: {combined.shape}")

            merged = merge_multiline_rows(combined, date_col=0, partic_col=2)
            merged.drop_duplicates(inplace=True)
            filtered = filter_valid_transactions(merged)

            df = (
                filtered
                .rename(columns={0: "date", 2: "description", 4: "withdrawal", 6: "deposit", 7: "balance"})
                [["date", "description", "withdrawal", "deposit", "balance", "page"]]
                .pipe(add_transaction_type)
                .pipe(add_amount_column)
            )

        elif bank_type == "axis bank":
            df = extract_table(pdf_file_path, flavor="stream", pages="all")
            if df is None or df.empty:
                df = extract_table(pdf_file_path, flavor="lattice", pages="all")
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="No data extracted from PDF.")
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

        parsed_data = df.to_dict(orient="records")
        with open("extracted_data.json", "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)

        return JSONResponse(status_code=200, content={"status": "success", "parsed_data": parsed_data})

    finally:
        os.remove(pdf_file_path)
