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
from collections import defaultdict

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
    pdf_bytes = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush()
    tmp.close()
    path = tmp.name

    try:
        bank = detect_bank_type(path)
        if bank != "jalgaon":
            return JSONResponse({"status":"unsupported bank type"})

        tables = camelot.read_pdf(path, flavor="lattice", pages="all", line_scale=50, split_text=False)
        if not tables:
            tables = camelot.read_pdf(path, flavor="stream", pages="all", strip_text="\n", edge_tol=200)

        # collect all table‐chunks per page
        pages = defaultdict(list)
        for t in tables:
            dfc = t.df.copy()
            dfc = fix_columns_for_page(dfc, t.page)
            dfc["page"] = t.page
            pages[t.page].append(dfc)

        # for each page, keep only the first chunk where neither col 5 nor col 7 is entirely blank
        clean_chunks = []
        for p, chunks in pages.items():
            for dfc in chunks:
                col5_blank = dfc[5].astype(str).eq("").all()
                col7_blank = dfc[7].astype(str).eq("").all()
                if not (col5_blank or col7_blank):
                    clean_chunks.append(dfc)
                    break  # only the first valid chunk per page

        if not clean_chunks:
            raise HTTPException(400, "No valid table found on any page")

        combined = pd.concat(clean_chunks, ignore_index=True)
        merged = merge_multiline_rows(combined, date_col=0, partic_col=2)
        merged.drop_duplicates(inplace=True)
        filtered = filter_valid_transactions(merged)

        # return for testing
        raw_filtered = filtered.to_dict(orient="records")
        return JSONResponse({"status": "filtered", "data": raw_filtered})

    finally:
        os.remove(path)
