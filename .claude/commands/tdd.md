---
description: Kent Beck의 TDD 원칙에 따라 Red-Green-Refactor 사이클 실행
---

# TDD (Test-Driven Development)

Kent Beck의 TDD 방법론에 따라 테스트 우선 개발을 진행합니다.

## 핵심 원칙 (CLAUDE.md)

1. **Red**: 실패하는 테스트를 먼저 작성
2. **Green**: 테스트를 통과하는 최소한의 코드 작성
3. **Refactor**: 테스트가 통과한 상태에서 코드 개선
4. **모든 주석은 한글**로 작성
5. **커밋은 테스트 통과 후에만**

## 프로젝트 특수 규칙 (PRD.md)

- **Data Leakage 절대 방지**: 미래 데이터가 입력에 포함되지 않도록 주의
- **Model Collapse 방지**: 단일 타겟만 사용 (다중 타겟 금지)
- **Tidy First**: 구조 변경과 동작 변경을 분리하여 커밋

## Instructions

### Phase 1: RED - 실패하는 테스트 작성

1. **plan.md 확인**:
   - `plan.md` 파일에서 다음 미완료 테스트 찾기
   - 파일이 없거나 테스트가 없으면 사용자에게 구현할 기능 질문

2. **테스트 작성**:
   - 실패하는 테스트를 **단 하나**만 작성
   - 테스트 이름은 동작을 명확히 설명 (예: `test_no_future_data_in_features`)
   - Given-When-Then 패턴 사용
   - 모든 주석은 **한글**로 작성
   - PRD 섹션 참조 주석 추가 (예: `# Data Leakage 방지 (PRD 2.1.2)`)

3. **테스트 실행**:
   - `pytest tests/test_xxx.py::test_name -v` 실행
   - 실패 확인 및 오류 메시지 분석
   - 사용자에게 실패 결과 보고

**예시**:
```python
# tests/test_data_leakage.py

def test_no_future_data_in_input():
    """
    입력 데이터에 미래 정보가 포함되지 않는지 검증

    Data Leakage 방지 (PRD 섹션 2.1.2)
    """
    # Given: 데이터셋 생성
    dataset = XRPTradingDataset(
        start_date=20250101,
        end_date=20250110,
        sequence_length=1440
    )

    # When: 특정 인덱스의 데이터 가져오기
    idx = 1500
    input_seq, target = dataset[idx]
    target_time = dataset.timestamps[idx + 1440]

    # Then: 모든 입력 타임스탬프가 타겟보다 과거여야 함
    input_times = dataset.timestamps[idx:idx+1440]
    assert all(t < target_time for t in input_times), \
        f"미래 데이터 발견: {max(input_times)} >= {target_time}"
```

### Phase 2: GREEN - 테스트 통과시키기

1. **최소한의 코드 작성**:
   - "가능한 가장 단순한 방법"으로 테스트 통과
   - 완벽한 설계보다 **작동하는 코드** 우선
   - 중복 코드 허용 (Refactor 단계에서 제거)
   - 모든 주석은 **한글**로 작성

2. **테스트 실행**:
   - 모든 테스트 실행 (새 테스트 + 기존 테스트)
   - `pytest tests/ -v` (전체) 또는 `pytest tests/test_xxx.py -v` (특정 파일)
   - **모든 테스트가 통과할 때까지 반복**

3. **결과 확인**:
   - 모든 테스트 통과 확인
   - 실패하면 코드 수정 후 재실행
   - 사용자에게 GREEN 상태 보고

**예시**:
```python
# utils/learning/data_generator_v2.py

class XRPTradingDataset(Dataset):
    def __init__(self, start_date, end_date, sequence_length):
        self.sequence_length = sequence_length
        self.data = self._load_data(start_date, end_date)
        self.timestamps = [doc['datetime'] for doc in self.data]

    def __getitem__(self, idx):
        """
        데이터 샘플 반환

        과거 데이터만 사용 (Data Leakage 방지)
        """
        # ✅ 입력: 과거 1440분 (idx부터 idx+1440 전까지)
        input_start = idx
        input_end = idx + self.sequence_length

        # ✅ 타겟: 미래 30분 (입력 이후)
        target_start = input_end
        target_end = input_end + 30

        input_seq = self._extract_features(self.data[input_start:input_end])
        target = self._extract_target(self.data[target_start:target_end])

        return input_seq, target
```

### Phase 3: REFACTOR - 코드 개선

**중요**: Refactor는 선택사항입니다. 개선할 부분이 없으면 생략 가능.

1. **구조 개선 검토**:
   - 중복 코드 제거
   - 변수/함수명 명확화
   - 함수 추출 (Extract Method)
   - 매직 넘버 상수화
   - 코드 가독성 향상

2. **각 리팩토링마다 테스트**:
   - **반드시** 한 번에 하나의 리팩토링만 수행
   - 각 변경 후 즉시 테스트 실행
   - 테스트 실패 시 즉시 되돌리기
   - **동작 변경 금지** (구조만 개선)

3. **Tidy First 원칙**:
   - 구조 변경은 별도 커밋
   - 동작 변경과 절대 혼합 금지

**예시**:
```python
# utils/learning/data_generator_v2.py

class XRPTradingDataset(Dataset):
    """
    XRP 거래 데이터셋

    Data Leakage 방지를 위해 과거 데이터만 사용 (PRD 2.1.2)
    """

    # 상수 추출
    DEFAULT_PREDICTION_HORIZON = 30

    def __init__(self, start_date, end_date, sequence_length,
                 prediction_horizon=DEFAULT_PREDICTION_HORIZON):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        self.data = self._load_data(start_date, end_date)
        self.timestamps = self._extract_timestamps(self.data)

        # 데이터 무결성 검증
        self._validate_data()

    def _validate_data(self):
        """데이터 검증 (Data Leakage 체크 포함)"""
        min_length = self.sequence_length + self.prediction_horizon
        if len(self.data) < min_length:
            raise ValueError(f"데이터 부족: {len(self.data)} < {min_length}")

    def __getitem__(self, idx):
        """데이터 샘플 반환"""
        # 메서드 추출로 의도 명확화
        input_indices = self._get_input_indices(idx)
        target_indices = self._get_target_indices(idx)

        input_seq = self._extract_features(self.data[input_indices])
        target = self._extract_target(self.data[target_indices])

        return input_seq, target

    def _get_input_indices(self, idx):
        """입력 데이터 인덱스 범위 계산 (과거만)"""
        return slice(idx, idx + self.sequence_length)

    def _get_target_indices(self, idx):
        """타겟 데이터 인덱스 범위 계산 (미래)"""
        start = idx + self.sequence_length
        return slice(start, start + self.prediction_horizon)
```

### Phase 4: COMMIT - 변경 사항 커밋

**커밋 규칙** (CLAUDE.md 엄수):

1. **커밋 조건 확인**:
   - [ ] 모든 테스트 통과
   - [ ] 컴파일러/린터 경고 없음
   - [ ] 단일 논리적 변경 단위

2. **Tidy First - 구조 변경 먼저**:
   ```bash
   git add utils/learning/validators.py
   git commit -m "structural: Extract validation logic

   - DataLeakageValidator 클래스 추가
   - 검증 로직 분리 (단일 책임 원칙)

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

3. **동작 변경 커밋**:
   ```bash
   git add tests/test_data_leakage.py utils/learning/data_generator_v2.py
   git commit -m "behavioral: Add data leakage prevention

   - 과거 데이터만 사용하도록 인덱싱 수정
   - 타임스탬프 검증 추가
   - Data Leakage 방지 테스트 추가

   Fixes: Data Leakage 위험 (PRD 8.1)

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

### Phase 5: NEXT - 다음 테스트

1. **plan.md 업데이트**:
   - 완료한 테스트를 `[x]`로 표시
   - 다음 테스트 확인

2. **사용자에게 질문**:
   - "다음 테스트를 진행하시겠습니까?"
   - "아니면 다른 작업을 하시겠습니까?"

3. **반복**:
   - Phase 1부터 다시 시작

## 특별 지침

### Data Leakage 방지 체크리스트

모든 테스트/코드 작성 시 확인:

- [ ] 모든 특성 계산이 `data[i-window:i]` 형태인가?
- [ ] `data[i:i+window]` 같은 미래 참조 없는가?
- [ ] Scaler fit이 Train 데이터로만 되었는가?
- [ ] 시간순 분할인가? (Random split 금지)

### Model Collapse 방지 체크리스트

- [ ] 타겟이 단일 특성인가?
- [ ] `close_log_diff`와 `tai_log_diff` 동시 사용 금지 확인
- [ ] Directional Accuracy 모니터링 설정했는가?

### 테스트 작성 가이드

**Good Test (좋은 테스트)**:
```python
def test_scaler_uses_train_data_only():
    """
    Scaler가 Train 데이터로만 학습되는지 검증

    Data Leakage 방지 (PRD 8.1)
    """
    # Given: Train/Val 데이터 분리
    train_data = load_data(20250101, 20250630)
    val_data = load_data(20250701, 20250831)

    # When: Scaler 생성
    scaler = create_scaler(train_data)

    # Then: Train 데이터 통계와 일치
    assert np.allclose(scaler.mean_, train_data.mean())

    # And: Val 데이터 포함 시 다른 값
    combined_mean = pd.concat([train_data, val_data]).mean()
    assert not np.allclose(scaler.mean_, combined_mean)
```

**Bad Test (나쁜 테스트)**:
```python
def test_stuff():  # ❌ 불명확한 이름
    # ❌ 주석 없음
    # ❌ Given-When-Then 구조 없음
    dataset = XRPTradingDataset(20250101, 20250110, 1440)
    assert len(dataset) > 0  # ❌ 의미 없는 검증
```

## 출력 형식

각 Phase 종료 시 다음 형식으로 보고:

```
✅ RED Phase 완료
------------------
테스트 파일: tests/test_data_leakage.py::test_no_future_data_in_input
상태: FAILED (예상된 실패)
오류: AssertionError: 미래 데이터 발견...

다음: GREEN Phase로 이동합니다.
```

```
✅ GREEN Phase 완료
-------------------
모든 테스트: PASSED (5/5)
변경 파일: utils/learning/data_generator_v2.py

다음: REFACTOR Phase로 이동하시겠습니까? (선택사항)
```

```
✅ REFACTOR Phase 완료
----------------------
리팩토링 내역:
- 메서드 추출: _get_input_indices, _get_target_indices
- 상수 추출: DEFAULT_PREDICTION_HORIZON
- 검증 로직 추가: _validate_data

모든 테스트: PASSED (5/5)

다음: COMMIT Phase로 이동합니다.
```

## 중요 사항

1. **절대 규칙**:
   - 테스트 없이 코드 작성 금지
   - 테스트 실패 중 커밋 금지
   - 구조/동작 변경 혼합 커밋 금지
   - 영어 주석 사용 금지 (한글만)

2. **Kent Beck의 격언**:
   - "Red-Green-Refactor 순서를 절대 바꾸지 마라"
   - "가능한 가장 단순한 것을 먼저 시도하라"
   - "테스트는 코드의 사양서다"
   - "작은 단계로 자주 커밋하라"

3. **실패 시 대응**:
   - 테스트 실패: 코드 수정 (Green Phase 반복)
   - 리팩토링 중 실패: 즉시 되돌리기
   - 막히면 더 작은 단계로 쪼개기
