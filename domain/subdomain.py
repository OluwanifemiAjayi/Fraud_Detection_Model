from pydantic import BaseModel

class PaymentRequest(BaseModel):
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

class PaymentResponse(BaseModel):
    isFraud: int