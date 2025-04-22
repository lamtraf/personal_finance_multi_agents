import os
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import asyncpg

# PostgreSQL connection URL
DATABASE_URL = os.getenv("DATABASE_URL")

print("DATABASE_URL:")
print(DATABASE_URL)

# DATABASE_URL = os.getenv("postgresql://kmoney-db:test1234@db:5432/kmoneydb")

async def get_connection():
    return await asyncpg.connect(DATABASE_URL)

from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime, UTC

class TransactionCreate(BaseModel):
    userId: str
    amount: float
    note: Optional[str] = None
    date: Optional[datetime] = None  # Optional if default used
    currencyId: str
    categoryId: str
    imageUrl: Optional[str] = None
    
class Category(BaseModel):
    id: str
    name: str
    
import uuid

async def create_transaction(data: TransactionCreate):
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
        return dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


async def get_categories() -> Dict[str, str]:
    query = """
        SELECT id, name FROM "Category"
    """
    conn = await get_connection()
    result = await conn.fetch(query)
    # return {
    #     "name": "id"
    # }
    return [Category(id=row["id"], name=row["name"]) for row in result]
    