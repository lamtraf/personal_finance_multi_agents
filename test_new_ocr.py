from utils import new_new_generate_ocr_table
import asyncio

async def main():
    urls = [
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2F0B18CE4B-BF2B-41C2-9AFE-E561458B076E.png?alt=media&token=f5020fee-9c20-4871-82d4-73f8303c05ed",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2F6856169E-9AF3-4A7A-907B-CB4B1DFDB332.png?alt=media&token=02c4027e-f91a-49f9-85f4-177a65dfb50d",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2F694526CB-3D70-407E-B6EB-07E9297E6906.png?alt=media&token=df34ca43-a515-4ae4-8bf8-124e558a8acf",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2F9965BA5C-1359-4303-919F-F70361C3FCDE.png?alt=media&token=45da5ac6-9274-40e3-941e-a41b1d8132cd",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2FAF5CCD8F-3C18-46E5-A89C-5E3E917257C4.png?alt=media&token=af589b15-95ae-4840-89ae-67dbece35b9a",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2FD60243BE-F7C8-4F67-A157-F33B32464582.png?alt=media&token=ab93c0a1-d486-45cb-a880-9be57a03c1de",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2FIMG_3830.jpeg?alt=media&token=13e01d47-be58-4e0f-a419-51b9895ab69b",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2FIMG_3918.jpg?alt=media&token=7aa160e8-718d-4557-9edd-8a72fad55384",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2FIMG_3919.jpg?alt=media&token=6839a27f-b1ac-4e9b-ad21-4021ef0dc19f",
       "https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2Ftest_receipt.jpg?alt=media&token=f049f85b-613b-473d-bb86-0c4137892d4a"
    ]
    
    for url in urls:
        print(f"Processing URL: {url}")
        result = await new_new_generate_ocr_table(url, "1", use_easyocr=False)
        print(result)
        print("-"*100)

if __name__ == "__main__":
    asyncio.run(main())
