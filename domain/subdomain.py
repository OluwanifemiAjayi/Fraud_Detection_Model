from pydantic import BaseModel

class PaymentRequest(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: int

class PaymentResponse(BaseModel):
    isFraud: int