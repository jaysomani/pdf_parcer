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
    if not is_date(str(row[0])):
        return False
    if str(row[3]) not in {"T", "L", "C"}:
        return False
    return True

def filter_valid_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.apply(is_valid_transaction, axis=1)].copy()

# (other helper functions unchanged…)

# -------------------------------------------
# FastAPI Endpoints
# -------------------------------------------
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

        if bank_type != "jalgaon":
            # (Your other branches remain unchanged…)
            pass

        # —— Jalgaon branch with robust lattice settings —— #
        # Attempt LATTICE first, with higher dpi & line_scale:
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

        # Fallback to STREAM if nothing useful:
        if not tables or len(tables) == 0:
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

        # Merge *all* chunks per page
        page_chunks = defaultdict(list)
        for tbl in tables:
            df_chunk = tbl.df.copy()
            df_fixed = fix_columns_for_page(df_chunk, tbl.page)
            page_chunks[tbl.page].append(df_fixed)

        if not page_chunks:
            raise HTTPException(400, "No valid tables extracted")

        # Concatenate each page’s chunks, then all pages
        combined = pd.concat(
            [pd.concat(chunks, ignore_index=True) for chunks in page_chunks.values()],
            ignore_index=True
        )
        print(f"[DEBUG] Combined shape before merge: {combined.shape}")

        # Now your existing merge/filter/rename logic:
        merged = merge_multiline_rows(combined, date_col=0, partic_col=2)
        merged.drop_duplicates(inplace=True)
        filtered = filter_valid_transactions(merged)

        df = (
            filtered
            .rename(columns={0: "date", 2: "description", 4: "withdrawal", 6: "deposit", 7: "balance"})
            [["date", "description", "withdrawal", "deposit", "balance"]]
            .pipe(add_transaction_type)
            .pipe(add_amount_column)
        )

        result = df.to_dict(orient="records")
        return JSONResponse(status_code=200, content={"status": "success", "parsed_data": result})

    finally:
        os.remove(pdf_file_path)
