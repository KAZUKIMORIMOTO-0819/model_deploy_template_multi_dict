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

def preprocess(df):
    df["feature_1"] = df["feature_1"]+1000
    return df