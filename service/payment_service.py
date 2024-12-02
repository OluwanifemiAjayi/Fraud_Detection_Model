import joblib
import pandas as pd

from xgboost import XGBClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domain.subdomain import PaymentRequest, PaymentResponse 

class PaymentService():
    def __init__(self): 
        self.path_model = "artifact/online_payment_fraud_model.pkl"

        self.path_scaler = "artifact/vector.pkl"
        self.model = self.load_artifact(self.path_model)
        self.scaler = self.load_artifact(self.path_scaler)

    def load_artifact(self, path_to_artifact):
        with open(path_to_artifact, 'rb') as f:
            artifact = joblib.load(f)
        return artifact
    
    def preprocess_input(self, request: PaymentRequest)->pd.DataFrame:
        data_dict = {
            "amount": request.amount,
            "oldbalanceOrg": request.oldbalanceOrg,
            "newbalanceOrig": request.newbalanceOrig,
            "oldbalanceDest": request.oldbalanceDest,
            "newbalanceDest": request.newbalanceDest,
            "type": request.type
            } 
        data_df = pd.DataFrame.from_dict([data_dict])

        columns_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        scaled_data = self.scaler.transform(data_df[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
        final_df = pd.concat([scaled_df.reset_index(drop=True), data_df[['type']].reset_index(drop=True)], axis=1)
        return final_df
    
    def predict_paymentintegrity(self, request: PaymentRequest)->PaymentResponse:
        input_df = self.preprocess_input(request)

        expected_order = self.model.get_booster().feature_names
        input_df = input_df[expected_order]

        payment_integrity = self.model.predict(input_df)[0]
        payment_integrity = int(payment_integrity)

        response = PaymentResponse
        response.isFraud = payment_integrity
        return response

# if __name__ == '__main__':
#    test_request = PaymentRequest(amount = -0.297555, oldbalanceOrg = -0.288654, newbalanceOrig = -0.292442, oldbalanceDest = -0.323814, newbalanceDest = -0.333411, type = 4)
    
#    pmt_serv = PaymentService()
#    result = pmt_serv.predict_paymentintegrity(request = test_request)
#    print(result.fraud)