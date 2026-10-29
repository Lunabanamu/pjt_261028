"""실습 07~08: 06번의 설정을 그대로 읽어 개발 16건으로 학습하고 test 4건을 최종 평가합니다."""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
#최종 평가에 사용할 여러가지 분류 평가 지표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

HERE = Path(__file__).resolve().parent #경로설정

# 제공: 06번을 먼저 완료하세요. 이 프로그램은 CV·설정 선택을 다시 하지 않습니다.
plan = json.loads((HERE / "selection.json").read_text(encoding="utf-8")) #06 결과를 가져옴
raw = pd.read_csv(HERE / "requests.csv").set_index("request_id")
develop = raw.loc[
    plan["train_ids"] + plan["valid_ids"] #train과 valid를 합친 후 최종 모델 학습에 이용
    ].copy() 
test = raw.loc[plan["test_ids"]].copy() 

# 제공: 행별 공식은 새 요청에도 동일하게 적용합니다. test로 통계나 설정을 학습하지 않습니다.
for table in (develop, test):
    table["log_input_chars"] = np.log1p(
        table["input_chars"]
        )
    
model = DecisionTreeClassifier(
    max_depth=plan["max_depth"], 
    random_state=plan["random_state"]
    )

# TODO B07a: 개발 자료 16건의 선택된 입력과 failed로 학습하세요. test는 넣지 않습니다.
model.fit( #기본구조 model.fit(X, y)
    develop[plan["features"]], 
    develop["failed"] #지도학습 y값
)

# 0·1 두 클래스가 있는 정상 자료입니다. classes_에서 양성 1의 위치를 확인합니다.
positive_column = model.classes_.tolist().index(1) #모델클래스
# TODO B07b: test의 실패 확률을 구하세요. predict_proba의 positive_column 열만 선택합니다.
probability = model.predict_proba(test[plan["features"]])[:, positive_column] #모델 클래스의 확률 반환 [:, 1]
"""predict()
→ 최종적으로 0인가 1인가?

predict_proba()
→ 0일 확률 / 1일 확률은 얼마인가?"""


prediction = (probability > plan["threshold"]).astype(int) #확률과 예측값은 다음

# TODO B07c: F1에는 0/1 예측값, AUC에는 실패 확률을 넣으세요.
f1 = f1_score(test["failed"], prediction, zero_division=0) #F1은 실제값+0/1예측값
auc = roc_auc_score(test["failed"], probability) #ROC-AUC는 실제값+실패 확률

metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1, "ROC_AUC": auc, "TP": int(((test["failed"] == 1) & (prediction == 1)).sum()),
           "actual_failed": int(test["failed"].sum()), "threshold": plan["threshold"]}
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
result.to_csv(HERE / "test_predictions.csv", index=False, encoding="utf-8-sig")
(HERE / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print("개발/최종평가:", len(develop), len(test))
print(result.round(3).to_string(index=False))
print("혼동행렬:", confusion_matrix(test["failed"], prediction, labels=[0, 1]).tolist())
print(f"Accuracy={metrics['Accuracy']:.3f}, Precision={metrics['Precision']:.3f}, Recall={metrics['Recall']:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
print("저장: test_predictions.csv, test_metrics.json")
print("08번은 이 설정 그대로 재실행합니다. test를 보고 06번 설정을 다시 고르지 않습니다.")


#결과보고
"""개발/최종평가: 16 4
request_id  actual  probability_failed  predicted
      B015       0               0.429          0
      B010       1               0.200          0
      B009       0               0.200          0
      B005       0               1.000          1
혼동행렬: [[2, 1], [1, 0]]
Accuracy=0.500, Precision=0.000, Recall=0.000, F1=0.000, AUC=0.167"""

#혼동행렬 TN=2 정상 2건/ FP=1 정상인데 실패 판단 1건/ FN=1 실패인데 정상 판단 1건/ TP=0 실패 맞힌 경우 0건
#Recall: 실제 실패를 하나도 찾지 못하여 0
#Precision: 실패라고 예측한 건 1건, 실제 실패가 아니었음 0
#F1: Precision과 Recall이 모두 0이므로 0

#Accuracy는 0.500이었지만 실제 실패를 하나도 탐지하지 못했기 때문에 실패 탐지 목적에서는 적절한 성능을 보이지 못함
#test 데이터가 4건으로 매우 적기 때문에, 이번 결과만으로 모델의 일반화 성능을 확정적으로 판단하기는 어려움