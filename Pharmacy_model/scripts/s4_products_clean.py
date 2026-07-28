import os
import json
import time
from groq import Groq  # <-- Thêm thư viện này

# --- 1. CẤU HÌNH API GROQ ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Sử dụng mã nguồn mở siêu nhanh của Meta
MODEL_NAME = "llama-3.1-8b-instant" 

# --- 4. HÀM GỌI API GROQ SIÊU NHANH VÀ AN TOÀN ---
def call_groq_api(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            # KHÔNG CẦN time.sleep(5.5) NỮA! Code sẽ chạy liên tục.
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                # Ép cấu hình trả về JSON Object giống như Gemini
                response_format={"type": "json_object"} 
            )
            
            res_content = response.choices[0].message.content
            parsed = json.loads(res_content)
            
            if isinstance(parsed, dict) and "questions" in parsed:
                return parsed["questions"]
            elif isinstance(parsed, list):
                return parsed
            else:
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
                return []
                
        except Exception as e:
            err_msg = str(e)
            # Xử lý khi quá hạn mức token (Rate limit tính theo Token Per Minute - TPM)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                print(f"\n⚠️ Groq Rate Limit. Nghỉ 20 giây... (Lần thử {attempt + 1}/{max_retries})")
                time.sleep(20)
            else:
                print(f"\n❌ Lỗi gọi API Groq: {err_msg}")
                time.sleep(2)
    return []