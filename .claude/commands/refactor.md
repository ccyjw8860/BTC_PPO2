---
description: TDD REFACTOR Phase - 테스트 유지하며 코드 개선
---

# REFACTOR Phase - 코드 개선

TDD의 세 번째 단계: 모든 테스트가 통과한 상태에서 코드 구조를 개선합니다.

## Instructions

1. **GREEN 상태 확인**:
   - 모든 테스트가 통과했는지 확인
   - 실패 중이면 REFACTOR 불가 경고

2. **리팩토링 후보 찾기**:
   - 중복 코드
   - 긴 함수 (> 20줄)
   - 매직 넘버
   - 불명확한 변수명
   - 복잡한 조건문

3. **한 번에 하나씩**:
   - **단 하나**의 리팩토링만 수행
   - 즉시 테스트 실행
   - 실패 시 즉시 되돌리기
   - 성공 시 다음 리팩토링

4. **Tidy First 원칙**:
   - **구조 변경만** (동작 변경 절대 금지)
   - 예시:
     - ✅ 함수명 변경
     - ✅ 메서드 추출
     - ✅ 상수 추출
     - ✅ 파일 이동
     - ❌ 알고리즘 변경
     - ❌ 새 기능 추가

5. **각 리팩토링 후 테스트**:
   ```bash
   pytest tests/ -v
   ```

6. **결과 보고**:
   - 수행한 리팩토링 목록
   - 최종 테스트 결과
   - 다음: 구조 변경 커밋 안내

## 리팩토링 카탈로그

### 1. Extract Method (메서드 추출)
```python
# Before
def process_data(data):
    # 검증
    if len(data) < 100:
        raise ValueError("데이터 부족")

    # 정규화
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std

    return normalized

# After
def process_data(data):
    self._validate_data(data)
    return self._normalize(data)

def _validate_data(self, data):
    """데이터 검증"""
    if len(data) < 100:
        raise ValueError("데이터 부족")

def _normalize(self, data):
    """정규화"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std
```

### 2. Extract Constant (상수 추출)
```python
# Before
if len(data) < 1440:
    raise ValueError("시퀀스 길이 부족")

# After
SEQUENCE_LENGTH = 1440

if len(data) < SEQUENCE_LENGTH:
    raise ValueError("시퀀스 길이 부족")
```

### 3. Rename (이름 변경)
```python
# Before
def calc(d):  # ❌ 불명확
    return d[0:1440]

# After
def get_input_sequence(data):  # ✅ 명확
    """입력 시퀀스 추출 (과거 1440분)"""
    return data[0:1440]
```

### 4. Remove Duplication (중복 제거)
```python
# Before
train_data = (data - np.mean(data)) / np.std(data)
val_data = (data - np.mean(data)) / np.std(data)

# After
def normalize(data):
    return (data - np.mean(data)) / np.std(data)

train_data = normalize(train_data)
val_data = normalize(val_data)
```

## 예시 출력

```
✅ REFACTOR Phase 완료
----------------------
리팩토링 1: Extract Method
  - _validate_data() 메서드 추출
  - 테스트: PASSED (5/5)

리팩토링 2: Extract Constant
  - SEQUENCE_LENGTH = 1440
  - DEFAULT_PREDICTION_HORIZON = 30
  - 테스트: PASSED (5/5)

리팩토링 3: Rename
  - get_data() → get_input_sequence()
  - 테스트: PASSED (5/5)

총 3개 리팩토링 완료
최종 테스트: PASSED (5/5)

다음 단계:
1. 구조 변경 커밋 (structural)
2. /commit_tdd 실행
```

## 주의사항

- **동작 변경 절대 금지**
- 테스트 결과가 바뀌면 안 됨
- 한 번에 하나씩만
- 실패 시 즉시 되돌리기
- 모든 주석은 **한글**

## Martin Fowler의 조언

"Refactoring is a controlled technique for improving the design of an existing code base."

리팩토링은 통제된 기법입니다:
- 작은 단계로
- 테스트로 검증
- 안전하게 개선
