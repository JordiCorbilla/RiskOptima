# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:53:30 2025

@author: jordi
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import datetime
import requests

app = FastAPI()

class Trade(BaseModel):
    ticker: str
    quantity: int
    price: float
    trade_time: Optional[str] = None  # ISO format

class TradeList(BaseModel):
    trades: List[Trade]

@app.post("/process_trades/")
def process_trades(data: TradeList):
    result = []
    for trade in data.trades:
        trade_value = trade.quantity * trade.price
        timestamp = trade.trade_time or datetime.datetime.utcnow().isoformat()
        result.append({
            "ticker": trade.ticker,
            "value": round(trade_value, 2),
            "timestamp": timestamp
        })
    return {"processed_trades": result}


import requests

response = requests.get("https://api.exchangerate.host/latest?base=GBP")
data = response.json()
print(data)
# Access nested fields
#eur_rate = data["rates"]["EUR"]
#print(f"Exchange rate GBP -> EUR: {eur_rate}")

