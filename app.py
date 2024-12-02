from fastapi import FastAPI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domain.subdomain import PaymentRequest, PaymentResponse 
from service.payment_service import PaymentService

fraud_detector_app = FastAPI()

@fraud_detector_app.post("/predict") 
async def predict_paymentintegrity(request: PaymentRequest)->PaymentResponse:
    return PaymentService().predict_paymentintegrity(request= request)