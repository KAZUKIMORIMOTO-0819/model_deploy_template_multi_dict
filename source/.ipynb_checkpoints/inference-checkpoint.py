import os
from io import BytesIO, StringIO
import lightgbm as lgb
import json
import logging
import json
import tarfile
import pandas as pd
import joblib
import boto3

from .preprocess import preprocess

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info("推論モデルをロードしています。")
    model = joblib.load(f"{model_dir}/model.joblib")
    logger.info("推論モデルをロード完了。")

    logger.info("label encoderモデルをロードしています。")
    label_encoders = joblib.load(f"{model_dir}/encoders.joblib")
    logger.info("label encoderモデルをロード完了。")
    return model, label_encoders

def input_fn(request_body, request_content_type,):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        ### dataframeに変換
        df1 = pd.DataFrame(input_data['data1'])
        df2 = pd.DataFrame(input_data["data2"])
        df = df1.copy()

        ### 前処理
        df = preprocess(df)
                
        return df
        
    elif request_content_type == 'text/csv':
        s = io.StringIO(request_body)
        df = pd.read_csv(s, header=None)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_and_encoders):
    model, label_encoders = model_and_encoders

    ### ラベルエンコーディングを適用 ※predict_fn()で行う必要がある。
    for column, encoder in label_encoders.items():
        if column in input_data.columns:
            input_data[column] = encoder.transform(input_data[column].astype(str))

    ### 推論
    prediction = model.predict(input_data)
    input_data["prediction"] = prediction

    # ### デコード
    # for column, encoder in label_encoders.items():
    #     if column in prediction.columns:
    #         input_data[column] = encoder.inverse_transform(input_data[column])
    
    return input_data, label_encoders

def output_fn(prediction_and_encoders, response_content_type):
    ### predictionとlabel_encodersを取得
    prediction, label_encoders = prediction_and_encoders
    
    ### デコード処理：各エンコードされたカラムをデコード
    for column, encoder in label_encoders.items():
        if column in prediction.columns:
            prediction[column] = encoder.inverse_transform(prediction[column])
    
    if response_content_type == 'application/json':
        ### S3に保存
        save_input_data_to_s3(input_data = prediction,
                              bucket_name = "sagemaker-ap-northeast-1-706711397653", 
                              object_key = "model_deploy_template_multi_dict/output_from_modelendpoint",
                             file_name = "output.csv")


        ### 返却する際はjson形式に直す必要がある。
        result = prediction.to_json(orient="records")  # DataFrameをJSON形式に変換
        return result
    elif response_content_type == 'text/csv':
        s = io.StringIO()
        pd.DataFrame(output).to_csv(s, index=False, header=False)
        return s.getvalue()
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

def change_output(prediction, ):
    return prediction*10000

def save_input_data_to_s3(input_data, bucket_name, object_key,file_name):
    """
    input_dataをS3にCSV形式で保存する関数。

    :param input_data: S3に保存するデータ (DataFrame形式)
    :param bucket_name: S3のバケット名
    :param object_key: S3上のオブジェクトキー（保存するファイルのパスと名前）
    """
    # S3クライアントの作成
    s3_client = boto3.client('s3')
    
    # DataFrameをCSV形式に変換
    csv_buffer = StringIO()
    input_data.to_csv(csv_buffer, index=False)
    
    # CSVデータをS3にアップロード
    s3_client.put_object(
        Bucket=bucket_name, 
        Key=os.path.join(object_key,file_name), 
        Body=csv_buffer.getvalue()
    )
    logger.info(f"Data saved to s3://{os.path.join(object_key,file_name)}")




