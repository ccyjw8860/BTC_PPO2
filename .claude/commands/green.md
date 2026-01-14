---
description: TDD GREEN Phase - 테스트를 통과시키는 최소 코드 작성
---

# GREEN Phase - 테스트 통과시키기

TDD의 두 번째 단계: 실패하는 테스트를 통과시키는 최소한의 코드만 작성합니다.

## Instructions

1. **RED 상태 확인**:
   - 실패하는 테스트가 있는지 확인
   - 없으면 `/red` 먼저 실행 요청

2. **최소 코드 작성**:
   - "가능한 가장 단순한 방법" 사용
   - 완벽한 설계 고려 안 함
   - 중복 코드 허용
   - 모든 주석은 **한글**
   - PRD 참조 주석 포함

3. **PRD 준수 확인**:
   - **Data Leakage**: `data[i-window:i]` 형태 확인
   - **Model Collapse**: 단일 타겟만 사용
   - **정규화**: Train 데이터로만 scaler fit

4. **테스트 실행**:
   ```bash
   # 전체 테스트 실행
   pytest tests/ -v

   # 또는 특정 파일만
   pytest tests/test_xxx.py -v
   ```

5. **결과 확인**:
   - 모든 테스트 PASSED 확인
   - 실패 시 코드 수정 후 재실행
   - 성공 시 GREEN 상태 보고

6. **다음 단계 안내**:
   - REFACTOR 필요 여부 질문
   - 또는 바로 COMMIT 제안

## 예시 출력

```
✅ GREEN Phase 완료
-------------------
변경 파일: utils/learning/models.py

추가 코드:
class ModelFactory:
    @staticmethod
    def create_model(output_size=1, **kwargs):
        """
        모델 생성

        Args:
            output_size: 출력 크기 (기본값: 1, 단일 타겟)

        Raises:
            ValueError: output_size > 1 시 (Model Collapse 방지, PRD 8.2)
        """
        if output_size > 1:
            raise ValueError(
                "단일 타겟만 지원합니다. "
                "다중 타겟은 Model Collapse를 유발합니다. "
                "PRD 섹션 8.2 참조"
            )
        # 모델 생성 로직
        return create_lstm_model(**kwargs)

테스트 결과: PASSED (5/5)

다음 단계:
- 리팩토링이 필요하면 /refactor 실행
- 바로 커밋하려면 /commit_tdd 실행
```

## Kent Beck의 조언

"Make it work, then make it right, then make it fast"
1. 먼저 작동하게 만들고 (GREEN) ← **현재 단계**
2. 그다음 올바르게 만들고 (REFACTOR)
3. 마지막으로 빠르게 만들어라 (OPTIMIZE)

## 주의사항

- 과도한 설계 금지 (YAGNI - You Aren't Gonna Need It)
- 테스트만 통과하면 충분
- 리팩토링은 다음 단계에서
