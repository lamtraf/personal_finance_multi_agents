import os
import traceback
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import asyncpg

# PostgreSQL connection URL
DATABASE_URL = os.getenv("DATABASE_URL")

print("DATABASE_URL: ", DATABASE_URL)

async def get_connection():
    return await asyncpg.connect(DATABASE_URL)

from pydantic import BaseModel
from typing import Dict, List, Optional, TypedDict
from datetime import datetime, UTC, timezone, timedelta

class TransactionCreate(BaseModel):
    userId: str
    amount: float
    note: Optional[str] = None
    date: Optional[datetime] = None
    currencyId: str
    categoryId: str
    imageUrl: Optional[str] = None
    
class Category(BaseModel):
    id: str
    name: str
    
import uuid

async def create_transaction(data: TransactionCreate)-> str:
    query = """
        INSERT INTO "Transaction" (
            id, "userId", amount, note, date, "currencyId",
            "categoryId", "createdAt", "updatedAt", "imageUrl"
        )
        VALUES (
            $1, $2, $3, $4, CURRENT_TIMESTAMP, $5,
            $6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, $7
        )
        RETURNING *
    """

    transaction_id = str(uuid.uuid4())

    try:
        conn = await get_connection()
        result = await conn.fetchrow(query,
            transaction_id,
            data.userId,
            data.amount,
            data.note,
            data.currencyId,
            data.categoryId,
            data.imageUrl
        )
        await conn.close()
        return result["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
async def insert_bulk_transactions(data: List[TransactionCreate]) -> List[str]:
    query = """
        INSERT INTO "Transaction" (
            id, "userId", amount, note, date, "currencyId",
            "categoryId", "createdAt", "updatedAt", "imageUrl"
        )
        VALUES (
            $1, $2, $3, $4, CURRENT_TIMESTAMP, $5,
            $6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, $7
        )
    """
    params = [
        (
            str(uuid.uuid4()),
            item.userId,
            item.amount,
            item.note,
            item.currencyId,
            item.categoryId,
            item.imageUrl
        )
        for item in data
    ]
    
    print("PARAMS: ", params)
    
    print("--------------------------------")
    
    try:
        conn = await get_connection()
        await conn.executemany(query, params)
        await conn.close()
        return [item[0] for item in params]
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    

class BudgetCreate(TypedDict):
    user_id: str
    amount: float 
    name: str
    category_id: str
    
async def insert_bulk_budgets(data: List[BudgetCreate]):
    query = """
        INSERT INTO "Budget" (
            id, "userId", amount, name, "categoryId", "createdAt", "updatedAt"
        )
        VALUES (
            $1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """
    params = [
        (
            str(uuid.uuid4()),
            item['user_id'],
            float(item['amount']),
            item['name'],
            item['category_id']
        )
        for item in data
    ]
    print("PARAMS: ", params)
    
    # insert to db
    try:
        conn = await get_connection()
        await conn.executemany(query, params)
        await conn.close()
        return [item[0] for item in params]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_categories() -> Dict[str, str]:
    query = """
        SELECT id, name FROM "Category"
    """
    conn = await get_connection()
    result = await conn.fetch(query)
    return [Category(id=row["id"], name=row["name"]) for row in result]
    