Personal Multi_Agents
Hiện đang có 2 Post đã hoạt động bình thường và đã test là /process_input và /process_sentiment_input. 
process_input sẽ lấy thông tin từ text, lưu vào database sqlite và respone về cho người dùng. 
process_sentiment thì sẽ phân tích cảm xúc qua text và đưa ra phản hồi phù hợp. 

** Hướng dẫn cài đặt
python -m venv venv
Windows: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser 
        venv\Scripts\Activate.ps1
MasOS: source venv/bin/activate
pip install -r requirements.txt

** Hướng dẫn tải llama3.2
- Tải Ollama và cài đặt
- Vào powershell, chạy lệnh ollama run llama3.2 để pull model về cho lần đầu tiên  và sau đó chạy bình thường
