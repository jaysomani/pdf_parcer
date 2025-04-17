#!/usr/bin/env python3

import os
import re
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

import pdfplumber
import camelot
import pandas as pd
import numpy as np

app = FastAPI()
handler = Mangum(app)

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPECTED_NCOLS = 8

def fix_columns_for_page(df: pd.DataFrame, page_num: int) -> pd.DataFrame:
    n = df.shape[1]
    if n < EXPECTED_NCOLS:
        for i in range(n, EXPECTED_NCOLS):
            df[i] = ""
    elif n > EXPECTED_NCOLS:
        df = df.iloc[:, :EXPECTED_NCOLS]
    df.columns = list(range(EXPECTED_NCOLS))
    return df

def is_date(s: str) -> bool:
    return bool(re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", s.strip()))

def merge_multiline_rows(df: pd.DataFrame, date_col=0, partic_col=2) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        if is_date(str(row[date_col])):
            out.append(row.copy())
        else:
            if out:
                out[-1][partic_col] = f"{out[-1][partic_col]} {row[partic_col]}"
    return pd.DataFrame(out, columns=df.columns)

def is_valid_transaction(row) -> bool:
    if not is_date(str(row[0]).strip()):
        return False
    return str(row[3]) in {"T", "L", "C"}

def filter_valid_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.apply(is_valid_transaction, axis=1)].copy()

def detect_bank_type(path: str) -> str:
    text = (pdfplumber.open(path).pages[0].extract_text() or "").lower()
    if "trndate valuedt particular insno / type withdrawals deposit balance" in text:
        return "jalgaon"
    return "unknown"

def add_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    def pick_type(r):
        w = float(r["withdrawal"] or 0)
        d = float(r["deposit"] or 0)
        if w > 0:
            return "payment"
        if d > 0:
            return "receipt"
        return "unknown"
    df["type"] = df.apply(pick_type, axis=1)
    return df

def add_amount_column(df: pd.DataFrame) -> pd.DataFrame:
    df["amount"] = df.apply(
        lambda r: float(r["withdrawal"]) if float(r["withdrawal"] or 0) > 0 else float(r["deposit"] or 0),
        axis=1
    )
    return df

def parse_balance(s: str) -> float:
    """Convert '2713.47Cr'→+2713.47 and '123.45Dr'→–123.45"""
    s = s.replace(",", "").strip()
    if s.endswith("Cr"):
        val = s[:-2]
        sign = 1
    elif s.endswith("Dr"):
        val = s[:-2]
        sign = -1
    else:
        val = s
        sign = 1
    try:
        return sign * float(val)
    except:
        return 0.0

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse("""
<!DOCTYPE html><html><body>
  <h1>Upload PDF</h1>
  <form enctype="multipart/form-data" method="post" action="/process-pdf">
    <label>Email: <input name="email" required></label><br>
    <label>Company: <input name="company" required></label><br>
    <input type="hidden" name="uploaded_file" value="uploaded.pdf">
    <input type="hidden" name="user_group" value="gold">
    <label>File: <input type="file" name="file" accept="application/pdf" required></label><br>
    <button type="submit">Submit</button>
  </form>
  <pre id="result"></pre>
  <script>
    document.querySelector("form").onsubmit = async e => {
      e.preventDefault();
      let res = await fetch("/process-pdf", { method:"POST", body:new FormData(e.target) });
      document.getElementById("result").innerText = JSON.stringify(await res.json(),null,2);
    };
  </script>
</body></html>
""")

@app.post("/process-pdf")
async def process_pdf(
    email: str = Form(...),
    company: str = Form(...),
    uploaded_file: str = Form("uploaded.pdf"),
    user_group: str = Form("gold"),
    file: UploadFile = File(...)
):
    # 1) save PDF
    pdf_bytes = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush()
    tmp.close()
    path = tmp.name

    try:
        # 2) detect bank
        bank = detect_bank_type(path)
        if bank != "jalgaon":
            return JSONResponse({"status":"unsupported bank type"})

        # 3) extract tables
        tables = camelot.read_pdf(path, flavor="lattice", pages="all", line_scale=50, split_text=False)
        if not tables:
            tables = camelot.read_pdf(path, flavor="stream", pages="all", strip_text="\n", edge_tol=200)

        # 4) first table per page
        page_tables = {}
        for t in tables:
            p = t.page
            if p not in page_tables:
                dfc = t.df.copy()
                dfc = fix_columns_for_page(dfc, p)
                dfc["page"] = p
                page_tables[p] = dfc

        if not page_tables:
            raise HTTPException(400, "No tables extracted")

        # 5) combine, merge, dedupe, filter
        combined = pd.concat(list(page_tables.values()), ignore_index=True)
        merged = merge_multiline_rows(combined, date_col=0, partic_col=2)
        merged.drop_duplicates(inplace=True)
        filtered = filter_valid_transactions(merged)

        # 6) fix blank balances by shifting
        mask = filtered[7].astype(str) == ""
        if mask.any():
            filtered.loc[mask, 7] = filtered.loc[mask, 6]
            filtered.loc[mask, 6] = filtered.loc[mask, 5]

        # 7) rename cols & keep
        df = filtered.rename(columns={
            0: "date", 2: "description", 4: "withdrawal",
            6: "deposit", 7: "balance"
        })
        df = df[["date","description","withdrawal","deposit","balance","page"]]

        # 8) type, amount
        df = add_transaction_type(df)
        df = add_amount_column(df)
        df = df[df["type"] != "unknown"].reset_index(drop=True)

        # 9) balance verification
        df["balance_num"] = df["balance"].apply(parse_balance)
        df["prev_balance"] = df["balance_num"].shift(1).fillna(df["balance_num"].iloc[0])
        df["expected_balance"] = df.apply(
            lambda r: r["prev_balance"] + (r["amount"] if r["type"]=="receipt" else -r["amount"]),
            axis=1
        )
        tol = 1e-2
        df["balance_match"] = (df["balance_num"] - df["expected_balance"]).abs() < tol
        df.loc[0, "balance_match"] = True  # first row always assumed OK

        # 10) return final JSON
        return JSONResponse({
            "status":"success",
            "receipts": df.to_dict(orient="records")
        })

    finally:
        os.remove(path)

