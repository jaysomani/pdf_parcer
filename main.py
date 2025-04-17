#!/usr/bin/env python3

import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import httpx

import pdfplumber
import camelot
import pandas as pd

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
        w = float(r.get("withdrawal", 0) or 0)
        d = float(r.get("deposit", 0) or 0)
        if w > 0:
            return "payment"
        if d > 0:
            return "receipt"
        return "unknown"
    df["type"] = df.apply(pick_type, axis=1)
    return df

def add_amount_column(df: pd.DataFrame) -> pd.DataFrame:
    df["amount"] = df.apply(
        lambda r: float(r["withdrawal"]) if float(r.get("withdrawal", 0) or 0) > 0 else float(r.get("deposit", 0) or 0),
        axis=1
    )
    return df

def parse_balance(s: str) -> float:
    s = s.replace(",", "").strip()
    if s.endswith("Cr"):
        val, sign = s[:-2], 1
    elif s.endswith("Dr"):
        val, sign = s[:-2], -1
    else:
        val, sign = s, 1
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
    <label>Temp Table: <input name="temp_table" required></label><br>
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
    temp_table: str = Form(...),
    uploaded_file: str = Form("uploaded.pdf"),
    user_group: str = Form("gold"),
    file: UploadFile = File(...)
):
    # Save PDF to temp
    pdf_bytes = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush()
    tmp.close()
    path = tmp.name

    try:
        # Detect bank
        if detect_bank_type(path) != "jalgaon":
            return JSONResponse({"status": "unsupported bank type"})

        # Extract tables via Camelot
        tables = camelot.read_pdf(path, flavor="lattice", pages="all", line_scale=50, split_text=False)
        if not tables:
            tables = camelot.read_pdf(path, flavor="stream", pages="all", strip_text="\n", edge_tol=200)

        # Keep only first table per page
        page_tables = {}
        for t in tables:
            if t.page not in page_tables:
                dfc = fix_columns_for_page(t.df.copy(), t.page)
                dfc["page"] = t.page
                page_tables[t.page] = dfc

        if not page_tables:
            raise HTTPException(400, "No tables extracted")

        # Combine, merge duplicates, filter
        combined = pd.concat(page_tables.values(), ignore_index=True)
        merged = merge_multiline_rows(combined, date_col=0, partic_col=2)
        merged.drop_duplicates(inplace=True)
        filtered = filter_valid_transactions(merged)

        # Fix blank balances
        mask = filtered[7].astype(str) == ""
        if mask.any():
            filtered.loc[mask, 7] = filtered.loc[mask, 6]
            filtered.loc[mask, 6] = filtered.loc[mask, 5]

        # Build entirechunk
        def build_chunk(r):
            parts = [str(r[c]) for c in range(EXPECTED_NCOLS)] + [str(r["page"])]
            return "".join(parts)
        filtered["entirechunk"] = filtered.apply(build_chunk, axis=1)

        # Rename/select final columns
        df = filtered.rename(columns={0: "date", 2: "description", 4: "withdrawal", 6: "deposit", 7: "balance"})
        df = df[["date", "description", "withdrawal", "deposit", "balance", "page", "entirechunk"]]

        # Add type, amount, balances
        df = add_transaction_type(df)
        df = add_amount_column(df)
        df = df[df["type"] != "unknown"].reset_index(drop=True)
        df["balance_num"] = df["balance"].apply(parse_balance)
        df["prev_balance"] = df["balance_num"].shift(1).fillna(df["balance_num"].iloc[0])
        df["expected_balance"] = df.apply(lambda r: r["prev_balance"] + (r["amount"] if r["type"]=="receipt" else -r["amount"]), axis=1)
        tol = 1e-2
        df["balance_match"] = (df["balance_num"] - df["expected_balance"]).abs() < tol
        df.loc[0, "balance_match"] = True

        # Prepare payload
        receipts = df.to_dict(orient="records")
        payload = {"temp_table": temp_table, "receipts": receipts}

        # Push to Node endpoint
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:3001/api/insertParsedReceipts",
                json=payload,
                timeout=30.0
            )
        if resp.status_code >= 300:
            return JSONResponse(status_code=resp.status_code, content=resp.json())

        # Return combined result
        return JSONResponse({
            "status":       "success",
            "temp_table":   temp_table,
            "insertResult": resp.json()
        })

    finally:
        os.remove(path)
