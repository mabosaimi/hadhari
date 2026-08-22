# Hadhari -  Arabic Spam Detection Model
Hadhari (حذارِ) is a machine learning model designed to detect spam messages in Arabic text. The name Hadhari comes from the Arabic word "حذارِ", meaning "beware" or "watch out". It was inspired by a professor who is known for saying this phrase so often in his AI class that it became his catch-phrase. The phrase seemed like the perfect fit for a model that helps users "beware" of spam!

**This model is particularly effective at detecting spam messages that contain:**
- Unsolicited advertisements targeting students, such as offers for cheap study materials, courses, or services.
- Fake excuses or fabricated claims for sick leave.

### Relevant Repositories:
#### Hadhari Whatsapp bot
https://github.com/mabosaimi/hadhari-wa

#### Hadhari website (Human-in-the-Loop interface)
https://github.com/mabosaimi/hadhari-ui

#### Hadhari Space (API)
https://huggingface.co/spaces/mabosaimi/hadhari/tree/main

#### Hadhari Model (Weights)
https://huggingface.co/mabosaimi/hadhari

---
**Prediction API Endpoint:** https://mabosaimi-hadhari.hf.space/predict

Expects a **POST** request with the JSON payload:
```json
{ "text": "الرسالة" }
```

**Response:**
```json
{
  "label": "not_spam",
  "class_id": 0,
  "confidence": 0.96
}
```
**Labels:**
- `spam` (Class 1)
- `not_spam` (Class 0)

**Confidence:**  
The model's confidence percentage in its prediction (0.0 to 1.0).


