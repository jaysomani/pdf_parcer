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
    allow_origins=["http://localhost:3000"],
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
            for table in tables:
                if len(table) < 2:
                    continue
                df = pd.DataFrame(table[1:], columns=table[0])
                df = deduplicate_columns(df)
                df = remove_newlines(df)
                df["page"] = page_num
                all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def detect_bank_type(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text() or ""
    text_lower = text.lower()
    patterns = {
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
    for pat, bank in patterns.items():
        if pat in text_lower:
            return bank
    if all(k in text_lower for k in ["transaction id", "txn posted date", "chequeno.", "transaction amount(inr)", "available balance(inr)"]):
        return "icici3"
    return "unknown"

def extract_icici3_with_pdfplumber(pdf_file):
    rows = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables():
                rows.extend(tbl)
    return pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()

def extract_table(pdf_file, flavor="stream", pages="all", **kwargs):
    tables = camelot.read_pdf(pdf_file, flavor=flavor, pages=pages, **kwargs)
    return pd.concat([t.df for t in tables], ignore_index=True) if tables else None

def merge_axis_rows(df, date_col="date"):
    def valid_date(val):
        return bool(re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", str(val).strip()))
    merged = []
    i = 0
    while i < len(df):
        row = df.iloc[i].copy()
        if valid_date(row[date_col]):
            j = i + 1
            while j < len(df) and not valid_date(df.iloc[j][date_col]):
                for col in df.columns:
                    if col != date_col:
                        row[col] = f"{row[col]} {df.iloc[j][col]}"
                j += 1
            merged.append(row)
            i = j
        else:
            i += 1
    return pd.DataFrame(merged, columns=df.columns)

def transform_sbi(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    req = ['txn date', 'description', 'debit', 'credit', 'balance']
    for col in req:
        if col not in df.columns:
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
    out = []
    for _, row in df.iterrows():
        dt = row['txn date'].strip()
        if not dt:
            continue
        desc = row['description'].strip()
        debit, credit = row['debit'] or 0, row['credit'] or 0
        if credit > 0:
            out.append({"txn_date": dt, "description": desc, "type": "receipt", "amount": credit, "ledger": ""})
        elif debit > 0:
            out.append({"txn_date": dt, "description": desc, "type": "payment", "amount": debit, "ledger": ""})
    return pd.DataFrame(out)

def normalize_identifier(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip())

# -------------------------------------------
# FastAPI Endpoints
# -------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse(content="""
    <!DOCTYPE html><html><head><title>PDF Test</title></head><body>
      <h1>Upload PDF</h1>
      <form enctype="multipart/form-data" method="post" action="/process-pdf">
        <label>Email:<input name="email" required></label><br>
        <label>Company:<input name="company" required></label><br>
        <input type="hidden" name="uploaded_file" value="uploaded.pdf">
        <input type="hidden" name="user_group" value="gold">
        <label>File:<input type="file" name="file" accept="application/pdf" required></label><br>
        <button type="submit">Go</button>
      </form>
      <pre id="result"></pre>
      <script>
        const form = document.querySelector("form");
        form.onsubmit = async e => {
          e.preventDefault();
          const res = await fetch("/process-pdf", {method:"POST", body:new FormData(form)});
          document.getElementById("result").innerText = JSON.stringify(await res.json(), null, 2);
        };
      </script>
    </body></html>
    """)

@app.get("/hello")
def hello():
    return {"message": "Hello, world!"}

@app.post("/echo-file")
async def echo_file(file: UploadFile = File(...)):
    data = await file.read()
    return {"filename": file.filename, "content_base64": base64.b64encode(data).decode()}

@app.post("/process-pdf")
async def process_pdf(
    email: str = Form(...),
    company: str = Form(...),
    uploaded_file: str = Form("uploaded.pdf"),
    user_group: str = Form("gold"),
    file: UploadFile = File(...)
):
    pdf_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name

    try:
        bank = detect_bank_type(path)
        print(f"[DEBUG] Detected bank type: {bank!r}")

        if bank == "unknown":
            return {"status": "unsupported bank type"}

        df = None

        if bank == "jalgaon":
            # lattice with stronger line detection
            tables = camelot.read_pdf(path, flavor="lattice", pages="all", line_scale=50, split_text=False)
            print(f"[DEBUG] Lattice tables: {len(tables)}")
            if not tables:
                print("[DEBUG] Falling back to stream")
                tables = camelot.read_pdf(path, flavor="stream", pages="all", strip_text="\n", edge_tol=200)
                print(f"[DEBUG] Stream tables: {len(tables)}")

            # merge chunks per page
            chunks = defaultdict(list)
            for t in tables:
                fixed = fix_columns_for_page(t.df.copy(), t.page)
                fixed["page"] = t.page
                chunks[t.page].append(fixed)

            if not chunks:
                raise HTTPException(400, "No tables extracted")

            combined = pd.concat([pd.concat(lst, ignore_index=True) for lst in chunks.values()], ignore_index=True)
            print(f"[DEBUG] Combined shape: {combined.shape}")

            merged = merge_multiline_rows(combined, date_col=0, partic_col=2)
            merged.drop_duplicates(inplace=True)
            filt = filter_valid_transactions(merged)

            df = (
                filt
                .rename(columns={0:"date", 2:"description", 4:"withdrawal", 6:"deposit", 7:"balance"})
                [["date","description","withdrawal","deposit","balance","page"]]
                .pipe(add_transaction_type)
                .pipe(add_amount_column)
            )

        # ... other bank branches unchanged ...

        if df is None or df.empty:
            raise HTTPException(400, "No data extracted")

        recs = df.to_dict(orient="records")
        with open("extracted_data.json","w",encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

        return {"status": "success", "parsed_data": recs}

    finally:
        os.remove(path)
