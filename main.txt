import tkinter as tk
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 데이터를 전처리하는 함수
def preprocess_data(data):
    data = np.array(data, dtype=float)
    return data

# ARIMA 모델을 생성하고 학습하는 함수
def create_and_train_model(data):
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    return model_fit

# 데이터 유효성 검사 함수
def validate_data(data):
    for i in data:
        if not i.isnumeric():
            return False
    return True

# 예측 기간을 사용자로부터 입력 받는 함수
def get_prediction_period():
    return int(prediction_period_entry.get())

# 예측값을 출력하는 함수
def predict():
    # 사용자 입력 데이터 전처리
    raw_data = entry.get().split(',')
    if not validate_data(raw_data):
        result_label.config(text='입력한 데이터가 유효하지 않습니다. 다시 입력해주세요.')
        return
    data = preprocess_data(raw_data)
    
    # ARIMA 모델 생성 및 데이터로 학습
    model_fit = create_and_train_model(data)
    
    # 사용자가 지정한 기간의 환자 가동 범위 예측
    prediction_period = get_prediction_period()
    output = model_fit.forecast(steps=prediction_period)
    predicted_values = output[0]

    # 결과 출력
    result_label.config(text=f'{prediction_period}일 후 환자의 가동 범위 예측값은 {predicted_values}입니다.')


# GUI 생성
root = tk.Tk()
root.title("환자 가동 범위 예측 프로그램")

label = tk.Label(root, text="날짜별 데이터를 ,로 구분하여 입력하세요.")
label.pack()

entry = tk.Entry(root)
entry.pack()

prediction_period_label = tk.Label(root, text="예측하고 싶은 기간을 입력하세요.")
prediction_period_label.pack()

prediction_period_entry = tk.Entry(root)
prediction_period_entry.pack()

button = tk.Button(root, text="예측하기", command=predict)
button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
