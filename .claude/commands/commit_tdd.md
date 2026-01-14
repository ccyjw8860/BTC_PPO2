---
description: TDD 커밋 - Tidy First 원칙에 따라 구조/동작 변경 분리 커밋
---

# TDD Commit - Tidy First 원칙 준수

Kent Beck의 "Tidy First" 원칙에 따라 구조 변경과 동작 변경을 분리하여 커밋합니다.

## Instructions

1. **커밋 전제 조건 확인**:
   - [ ] 모든 테스트 PASSED
   - [ ] 린터/컴파일러 경고 없음
   - [ ] 단일 논리적 변경 단위

2. **변경 사항 분석**:
   - Git status 확인
   - 구조 변경 vs 동작 변경 분류

3. **Tidy First - 구조 변경 먼저 커밋**:

   **구조 변경 (Structural Changes)**:
   - 함수/변수명 변경
   - 메서드 추출/인라인
   - 파일 이동
   - 상수 추출
   - 중복 제거

   ```bash
   git add utils/learning/validators.py
   git commit -m "structural: Extract validation logic

   - DataLeakageValidator 클래스 추가
   - 검증 로직 분리 (단일 책임 원칙)
   - 재사용성 향상

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

4. **동작 변경 커밋**:

   **동작 변경 (Behavioral Changes)**:
   - 새 기능 추가
   - 버그 수정
   - 알고리즘 변경
   - 테스트 추가

   ```bash
   git add tests/test_data_leakage.py utils/learning/data_generator_v2.py
   git commit -m "behavioral: Add data leakage prevention

   - 과거 데이터만 사용하도록 인덱싱 수정
   - 타임스탬프 검증 로직 추가
   - Data Leakage 방지 테스트 추가

   Fixes: Data Leakage 위험 (PRD 8.1)

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

5. **plan.md 업데이트** (선택사항):
   - 완료한 테스트를 `[x]`로 표시
   - 다음 테스트 확인

6. **다음 단계 안내**:
   "다음 TDD 사이클을 시작하시겠습니까? /tdd 또는 /red 실행"

## 커밋 메시지 템플릿

### 구조 변경 (structural)
```
structural: <간단한 제목>

- <변경 사항 1>
- <변경 사항 2>
- <변경 사항 3>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 동작 변경 (behavioral)
```
behavioral: <간단한 제목>

- <변경 사항 1>
- <변경 사항 2>
- <변경 사항 3>

Fixes: <해결한 문제> (PRD <섹션>)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 버그 수정 (fix)
```
fix: <버그 설명>

- <수정 내용 1>
- <수정 내용 2>

Problem: <문제 상황>
Solution: <해결 방법>

Fixes: PRD <섹션>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## 예시 출력

```
✅ 커밋 전제 조건 확인
-----------------------
테스트: PASSED (5/5)
린터: OK
워킹 디렉토리: 3 files changed

변경 파일:
  M utils/learning/validators.py          (NEW - 구조 변경)
  M utils/learning/data_generator_v2.py   (동작 변경)
  M tests/test_data_leakage.py            (NEW - 동작 변경)

✅ 커밋 1/2 - 구조 변경
-----------------------
git add utils/learning/validators.py
git commit -m "structural: Extract DataLeakageValidator class
...
[main abc1234] structural: Extract DataLeakageValidator class
 1 file changed, 45 insertions(+)

✅ 커밋 2/2 - 동작 변경
-----------------------
git add tests/test_data_leakage.py utils/learning/data_generator_v2.py
git commit -m "behavioral: Add data leakage prevention
...
[main def5678] behavioral: Add data leakage prevention
 2 files changed, 78 insertions(+), 12 deletions(-)

✅ plan.md 업데이트
------------------
[x] Data Leakage 방지 테스트
[ ] Model Collapse 방지 테스트

모든 커밋 완료!

다음 단계: /tdd 또는 /red 로 다음 테스트 시작
```

## 절대 규칙

❌ **절대 하지 말 것**:
- 구조 변경과 동작 변경을 한 커밋에 섞기
- 테스트 실패 중 커밋
- 린터 경고 무시하고 커밋
- 의미 없는 커밋 메시지 ("fix", "update" 등)

✅ **반드시 할 것**:
- 구조 변경 먼저 커밋
- 동작 변경 나중에 커밋
- 각 커밋은 논리적 단위
- PRD 섹션 참조 명시

## Kent Beck의 격언

"Commit early, commit often"
- 작은 단위로
- 자주 커밋
- 원자적 변경
- 명확한 메시지
