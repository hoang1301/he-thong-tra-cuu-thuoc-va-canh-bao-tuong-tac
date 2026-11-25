from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR


def ocr(image_path: str) -> str:
    engine = RapidOCR()
    result, _ = engine(image_path)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def normalize_drug_name(text: str) -> str:
    client = OpenAI(
        base_url="http://localhost:3001/v1",
        api_key="freellmapi-b7079f0480566e7a5a4f784cbd933a48d68bc2d0a211aac1",
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": (
                    "Bạn là một chuyên gia y tế chuyên bóc tách dữ liệu OCR. Hãy tìm và chỉ in ra tên "
                    "thương mại của thuốc từ đoạn văn bản sau. Nếu tên thuốc bị viết sai chính tả, mất dấu, "
                    "hoặc dính chữ do lỗi quét OCR, hãy tự động sửa lại và chuẩn hóa về dạng tên thuốc đúng "
                    "chính xác trên thực tế. Không bao gồm hoạt chất, không giải thích, không chào hỏi, "
                    f"không thêm bất kỳ ký tự nào khác ngoài tên thuốc đã chuẩn hóa: {text}"
                ),
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    image_path = "photo_6239982346425798374_y.jpg"
    raw_text = ocr(image_path)
    print("--- KẾT QUẢ OCR THÔ ---")
    print(raw_text if raw_text else "(Không đọc được chữ)")

    drug_name = normalize_drug_name(raw_text)
    print("--- TÊN THUỐC CHUẨN HÓA ---")
    print(drug_name)