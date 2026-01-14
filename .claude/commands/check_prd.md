---
description: PRD 위험 요소 자동 체크 (Data Leakage, Model Collapse 등)
---

# PRD 체크 - 위험 요소 자동 검사

PRD에 명시된 위험 요소를 코드베이스에서 자동으로 검사합니다.

## Instructions

1. **전체 코드베이스 스캔**:
   - `utils/` 디렉토리 우선
   - `tests/` 디렉토리
   - 루트 Python 파일

2. **Data Leakage 패턴 검색** (PRD 8.1):

   **HIGH 위험**:
   - `scaler.fit(entire_dataset)` 또는 `fit(data)`
   - `data[i:i+n]` (미래 참조 인덱싱)
   - `train_test_split(..., shuffle=True)`
   - `fillna(method='bfill')` 또는 `backward_fill()`

   **MEDIUM 위험**:
   - `rolling(..., center=True)`
   - `shift(-n)` (음수 shift)
   - `data.loc[start:end]` (경계 확인 필요)

3. **Model Collapse 패턴 검색** (PRD 8.2):

   **HIGH 위험**:
   - `target_columns = ['close_log_diff', 'tai_log_diff']`
   - `output_size > 1` 또는 `prediction_horizon` 여러 타겟
   - `loss = mse_loss` 단독 (Directional Accuracy 없음)

   **MEDIUM 위험**:
   - 상관관계 높은 특성 동시 사용
   - Multi-task learning without validation

4. **일반 안티패턴 검색**:
   - 영어 주석 (한글 주석 규칙 위반)
   - 매직 넘버 (상수화 필요)
   - 긴 함수 (> 50줄)

5. **결과 리포트 생성**:

```
🔍 PRD 위험 요소 검사 결과
============================

📊 총 발견: 5개 (HIGH: 2, MEDIUM: 3, LOW: 0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 HIGH 위험 (2개)
------------------

1. Data Leakage: 전체 데이터셋 정규화
   파일: utils/learning/data_generator_v2.py:45
   코드: scaler.fit(entire_dataset)

   문제: Train + Val + Test 데이터 모두 사용
   해결: scaler.fit(train_dataset)으로 변경

   참조: PRD 섹션 8.1, 부록 C 패턴 1

2. Model Collapse: 다중 타겟 사용
   파일: utils/learning/models.py:78
   코드: target_columns = ['close_log_diff', 'tai_log_diff']

   문제: 상관관계 높은 두 타겟 동시 사용
   해결: target_columns = ['returns'] 단일 타겟 사용

   참조: PRD 섹션 8.2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  MEDIUM 위험 (3개)
---------------------

3. Data Leakage: 미래 참조 가능성
   파일: utils/features/create_ta.py:123
   코드: df.shift(-5)

   문제: 음수 shift는 미래 데이터 참조
   해결: df.shift(5)로 변경 (과거 참조)

   참조: PRD 부록 A

4. 영어 주석 사용
   파일: utils/learning/data_generator_v2.py:67
   코드: # Calculate moving average

   문제: CLAUDE.md 규칙 위반 (한글 주석)
   해결: # 이동 평균 계산

   참조: CLAUDE.md

5. Directional Accuracy 모니터링 누락
   파일: utils/learning/trainer.py:156
   코드: metrics = {'loss': mse_loss, 'mae': mae}

   문제: Model Collapse 조기 감지 불가
   해결: 'directional_accuracy' 메트릭 추가

   참조: PRD 부록 B

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 권장 사항
------------

우선순위 1 (즉시 수정):
- Data Leakage 패턴 2개 수정
- Model Collapse 위험 1개 수정

우선순위 2 (가능한 빨리):
- Directional Accuracy 추가
- 영어 주석 한글 변환

우선순위 3 (시간 날 때):
- 코드 품질 개선

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 참고 문서
------------
- PRD 섹션 8: 위험 요소 및 대응 방안
- PRD 부록 A: Data Leakage 방지 체크리스트
- PRD 부록 B: Model Collapse 방지 체크리스트
- PRD 부록 C: 일반적인 Data Leakage 패턴
- CLAUDE.md: TDD 및 코드 품질 규칙
```

## 검사 패턴 상세

### Data Leakage 패턴

```python
# ❌ HIGH: 전체 데이터 정규화
scaler.fit(df)
scaler.fit(entire_dataset)
scaler.fit(pd.concat([train, val, test]))

# ❌ HIGH: 미래 데이터 사용
features[i] = data[i:i+60].mean()
df.loc[start:future_date]

# ❌ HIGH: Random shuffle
train_test_split(data, shuffle=True)
df.sample(frac=1)

# ❌ MEDIUM: 센터링 rolling
df.rolling(window=20, center=True)

# ✅ SAFE
scaler.fit(train_only)
features[i] = data[i-60:i].mean()
train, val = time_series_split(data)
```

### Model Collapse 패턴

```python
# ❌ HIGH: 다중 타겟
target_columns = ['close_log_diff', 'tai_log_diff']
output_size = 2

# ❌ MEDIUM: Directional Accuracy 없음
metrics = {'loss': mse, 'mae': mae}

# ✅ SAFE
target_columns = ['returns']
output_size = 1
metrics = {'loss': mse, 'directional_accuracy': dir_acc}
```

## 자동 수정 제안

HIGH 위험 항목에 대해 자동 수정 코드 제안:

```python
# 수정 전
scaler.fit(entire_dataset)

# 수정 후 (제안)
# Train 데이터로만 scaler 학습 (Data Leakage 방지, PRD 8.1)
scaler.fit(train_dataset)
```

## 주의사항

- False positive 가능 (수동 확인 필요)
- 컨텍스트에 따라 안전할 수 있음
- 최종 판단은 개발자가 수행
