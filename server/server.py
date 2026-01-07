# server.py (Đã sửa lỗi CUDA và kết nối)
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class ServerConfig:
    model_name: str = "distilbert-base-uncased"
    model_path: str = "./best_distilbert_model1.bin"
    max_len: int = 128
    # THAY ĐỔI 1: Ép về CPU để tránh lỗi "CUDA error: no kernel image"
    device: str = "cpu" 
    allow_origins: tuple[str, ...] = ("*",) 
    threshold: float = 0.8984
# =========================
# DTO (Request Model)
# =========================
class URLInput(BaseModel):
    url: str

# =========================
# SERVICE: Model Inference
# =========================
class PhishingModelService:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        # THAY ĐỔI 2: Luôn sử dụng CPU để đảm bảo tính tương thích tuyệt đối
        self.device = torch.device("cpu")

        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.model: Optional[DistilBertForSequenceClassification] = None
        self._ready: bool = False
        self._load_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Load tokenizer + base model, then load fine-tuned weights."""
        try:
            print(f"Server: Đang nạp mô hình lên {self.device}...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.cfg.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.cfg.model_name,
                num_labels=2
            )

            # Nạp trọng số và ánh xạ thẳng vào CPU
            state = torch.load(self.cfg.model_path, map_location=self.device)
            self.model.load_state_dict(state)

            self.model.to(self.device)
            self.model.eval()

            self._ready = True
            self._load_error = None
            print("Server: DistilBERT đã sẵn sàng trên CPU.")
        except Exception as e:
            self._ready = False
            self._load_error = f"Lỗi nạp mô hình: {e}"
            print(f"LỖI SERVER: {self._load_error}")

    def predict(self, url: str) -> Dict[str, Any]:
        if not self.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError(self._load_error or "Mô hình chưa sẵn sàng.")

        encoding = self.tokenizer.encode_plus(
            url,
            max_length=self.cfg.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs_tensor = torch.softmax(outputs.logits, dim=1)
            score = float(probs_tensor.squeeze()[1].item())

        is_phishing = score >= self.cfg.threshold

        return {
            "url": url,
            "is_phishing": is_phishing,
            "score": round(score, 4),
            "message": "CẢNH BÁO" if is_phishing else "An toàn",
        }

# =========================
# APP FACTORY
# =========================
class ApiApplication:
    def __init__(self, cfg: ServerConfig, service: PhishingModelService):
        self.cfg = cfg
        self.service = service
        self.app = FastAPI()
        self._setup_cors()
        self._setup_routes()
        self._setup_startup()

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=list(self.cfg.allow_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_startup(self) -> None:
        @self.app.on_event("startup")
        async def _on_startup():
            self.service.load()

    def _setup_routes(self) -> None:
        @self.app.post("/predict")
        async def predict_phishing(data: URLInput):
            try:
                return self.service.predict(data.url)
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {e}")

def create_app() -> FastAPI:
    cfg = ServerConfig()
    service = PhishingModelService(cfg)
    api = ApiApplication(cfg, service)
    return api.app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # THAY ĐỔI 3: Đổi thành 0.0.0.0 để Cloudflare Tunnel có thể truy cập
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
