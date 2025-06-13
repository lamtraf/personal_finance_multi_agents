import asyncio
from new_agent import invoke_graph_stream, get_graph


initial_extractor_state = {
    "user_id": "68aaed9e-b3b3-481a-a30d-5cec3e248dc7",
    "user_input": "Mua gà rán 500k",
    "image_url": None
}

initial_question_state = {
    "user_id": "68aaed9e-b3b3-481a-a30d-5cec3e248dc7",
    "user_input": "Tháng vừa rồi tiêu bao nhiêu tiền?",
    "image_url": None
}

initial_ocr_state = {
    "user_id": "26138326-d395-4081-a154-f8955d489f7a",
    "user_input": None,
    "image_url": r"https://firebasestorage.googleapis.com/v0/b/k-money-stg.firebasestorage.app/o/transactionReceipts%2F0B18CE4B-BF2B-41C2-9AFE-E561458B076E.png?alt=media&token=f5020fee-9c20-4871-82d4-73f8303c05ed"
}

async def main():
    # async for chunk in invoke_graph_stream(initial_extractor_state):
        # print(chunk)
    
    # result = await new_advice_workflow.ainvoke(initial_advice_state)
    
    # print(result)
    # result = await new_extractor_graph.ainvoke(initial_state)
    # print(result)
    
    print(get_graph())

if __name__ == "__main__":
    asyncio.run(main())


