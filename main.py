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
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
handler = Mangum(app)

# -------------------------------------------
# Helpers
# -------------------------------------------

def add_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    def get_type(row):
        d = float(str(row["deposit"]).replace(",", "") or 0)
        w = float(str(row["withdrawal"]).replace(",", "") or 0)
        if d > 0:
            return "receipt"
        elif w > 0:
            return "payment"
        else:
            return "unknown"
    df["type"] = df.apply(get_type, axis=1)
    return df

def add_amount_column(df: pd.DataFrame) -> pd.DataFrame:
    df["amount"] = df.apply(
        lambda r: r["deposit"] if r["type"] == "receipt"
                  else (r["withdrawal"] if r["type"] == "payment" else None),
        axis=1
    )
    return df[["date","description","balance","type","amount"]]

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
    if not is_date(str(row[0])): 
        return False
    return str(row[3]) in {"T","L","C"}

def filter_valid_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.apply(is_valid_transaction, axis=1)].copy()

def detect_bank_type(path: str) -> str:
    text = (pdfplumber.open(path).pages[0].extract_text() or "").lower()
    patterns = {
        "trndate valuedt particular insno / type withdrawals deposit balance": "jalgaon",
        # … other bank patterns …
    }
    for pat, name in patterns.items():
        if pat in text:
            return name
    return "unknown"

# -------------------------------------------
# FastAPI
# -------------------------------------------

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
    # save temp PDF
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(data); tmp.flush(); tmp.close()
    path = tmp.name

    try:
        bank = detect_bank_type(path)
        print(f"[DEBUG] Detected bank type: {bank!r}")
        if bank != "jalgaon":
            return JSONResponse({"status":"unsupported bank type"})

        # try lattice with stronger line detection
        tables = camelot.read_pdf(path, flavor="lattice", pages="all", line_scale=50, split_text=False)
        if not tables:
            # fallback to stream
            tables = camelot.read_pdf(path, flavor="stream", pages="all", strip_text="\n", edge_tol=200)

        # merge chunks per page
        pages = defaultdict(list)
        for t in tables:
            dfc = t.df.copy()
            dfc = fix_columns_for_page(dfc, t.page)
            dfc["page"] = t.page
            pages[t.page].append(dfc)

        if not pages:
            raise HTTPException(400, "No tables extracted")

        combined = pd.concat(
            [pd.concat(chs, ignore_index=True) for chs in pages.values()],
            ignore_index=True
        )

        # ==== EARLY RETURN FOR TESTING: raw combined data ==== #
        raw = combined.to_dict(orient="records")
        return JSONResponse({"status": "raw", "data": raw})

        # ==== Rest of processing (currently skipped) ==== #
        # merged = merge_multiline_rows(combined, date_col=0, partic_col=2)
        # merged.drop_duplicates(inplace=True)
        # valid = filter_valid_transactions(merged)
        # df = valid.rename(columns={0:"date",2:"description",4:"withdrawal",6:"deposit",7:"balance"})
        # df = df[["date","description","withdrawal","deposit","balance","page"]]
        # df = add_transaction_type(df)
        # df = add_amount_column(df)
        # df = df[df["type"]!="unknown"]
        # records = df.to_dict("records")
        # return JSONResponse({"status":"success","parsed_data":records})

    finally:
        os.remove(path)
