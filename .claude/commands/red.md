---
description: TDD RED Phase - 실패하는 테스트만 작성
---

# RED Phase - 실패하는 테스트 작성

TDD의 첫 번째 단계: 실패하는 테스트 작성에만 집중합니다.

## Instructions

1. **사용자에게 질문**:
   "어떤 기능을 테스트하시겠습니까?"

2. **plan.md 확인** (선택사항):
   - `plan.md` 파일 존재 시 다음 미완료 테스트 제안
   - 없으면 사용자 입력 기반으로 진행

3. **테스트 작성**:
   - **단 하나**의 실패하는 테스트만 작성
   - Given-When-Then 패턴 사용
   - 모든 주석은 **한글**
   - PRD 참조 주석 포함

4. **PRD 위험 요소 체크**:
   - Data Leakage 가능성 검토
   - Model Collapse 가능성 검토
   - 해당 사항 있으면 테스트에 검증 로직 추가

5. **테스트 실행**:
   ```bash
   pytest tests/test_xxx.py::test_name -v
   ```

6. **결과 보고**:
   - 실패 확인 (RED 상태)
   - 오류 메시지 분석
   - 다음 단계: `/green` 명령어 실행 안내

## 예시 출력

```
✅ RED Phase 완료
------------------
테스트 파일: tests/test_model_collapse.py::test_single_target_only
테스트 코드:

def test_single_target_only():
    """
    단일 타겟만 허용 (Model Collapse 방지)

    PRD 섹션 8.2 참조
    """
    # Given: 다중 타겟 설정
    config = {'output_size': 2}  # close_log_diff + tai_log_diff

    # When & Then: ValueError 발생 확인
    with pytest.raises(ValueError, match="단일 타겟만 지원"):
        model = ModelFactory.create_model(**config)

실행 결과: FAILED
오류: test_model_collapse.py:10: 예상된 ValueError가 발생하지 않음

다음 단계: /green 명령어로 코드를 작성하여 테스트를 통과시키세요.
```

## 주의사항

- GREEN Phase로 넘어가지 마세요 (테스트만 작성)
- 코드 구현은 하지 마세요
- 한 번에 하나의 테스트만 작성
