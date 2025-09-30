# Repository Guidelines

## Project Structure & Module Organization
Work from `account-tax/`, the active Kedro workspace. Pipelines sit in `src/account_tax/pipelines/` (ingestion, preprocess, feature, split, train) and register through `pipeline_registry.py`. Utilities belong in `src/account_tax/utils/`. Keep shared config in `conf/base/*.yml` and secrets in untracked `conf/local/`. Dataset artifacts follow the Kedro layer folders `data/01_raw`…`data/08_reporting`; commit only lightweight fixtures. Mirror package modules with tests under `tests/`.

## Environment & Setup
Standardise on Python 3.12. Create a virtualenv (`python -m venv .venv && source .venv/bin/activate`) and install dependencies with `pip install -r account-tax/requirements.txt`. Add developer tooling via `pip install -e account-tax[dev]`. Ensure the `kedro` CLI is available so notebook and pipeline commands run reliably.

## Build, Test, and Development Commands
- `cd account-tax && kedro run` — execute the default pipeline through the split stage.
- `cd account-tax && kedro run --pipeline full` — run end-to-end training; outputs land in `data/05_model_input/` and beyond.
- `cd account-tax && pytest -q` — run automated tests; append `--cov=account_tax --cov-report=term-missing` for coverage.
- `cd account-tax && kedro viz --host 0.0.0.0 --port 4141` — inspect the pipeline graph.

## Coding Style & Naming Conventions
Format code with Black (4-space indentation, 88-character lines) and run `black account-tax/src account-tax/tests` before submitting. Use `isort` for imports and `flake8` for linting. Adopt `snake_case` for modules, functions, and Kedro node names, reserving `PascalCase` for classes. Provide type hints and succinct docstrings, as in `src/account_tax/utils/label_manager.py`.

## Testing Guidelines
Add tests in `account-tax/tests/`, mirroring the package you touch (for example `tests/pipelines/feature/test_nodes.py`). Prefer `pytest` fixtures for catalog stubs instead of mutating real data. New nodes need behavioural coverage, and new pipelines require a smoke test that loads the Kedro catalog. Treat coverage thresholds as part of the review checklist.

## Commit & Pull Request Guidelines
Git history is not yet established, so adopt Conventional Commits immediately (e.g., `feat: add split pipeline smoke test`, `fix: handle missing dummy labels`). Keep commits focused and reference affected pipelines or datasets in the body. PRs should state intent, list validation commands (`kedro run`, `pytest`), and flag configuration or catalog changes. Share `kedro viz` screenshots when altering pipeline topology and link relevant issues.

## Data & Configuration Practices
Do not commit credentials or large datasets; store secrets in `conf/local/` or environment variables. Update `conf/base/catalog.yml` and `parameters.yml` alongside code that introduces new datasets or parameters. When sample inputs are required, add sanitised fixtures under `data/01_raw/` and note their provenance in the PR.

## 에이전트 역할 가이드

### architecture (설계자)
- 철학: **대칭화(패턴화)**, **모듈화(노드 기반 구조)**, **순서화(인과 정렬)**를 모든 설계 판단의 최우선 기준으로 삼습니다.
  - 대칭화: 동일한 본질의 함수·파이프라인은 유사한 패턴으로 작성돼야 하며, 불필요한 뇌자원 소비 없이 구조를 파악할 수 있게 합니다.
  - 모듈화: 노드 단위로 기능을 분리하고, `pipeline.py`를 통해 일관된 연결 패턴을 유지합니다.
  - 순서화: 정적인 폴더/파일 구조와 동적인 실행 순서를 모두 문서화해 인과 관계를 명확히 합니다.
- 기본 설계도는 `폴더 경로 → 파일명 → 함수` 순으로 정의하며, 각 파일은 동일한 역할을 수행하는 함수 블록만 포함해야 합니다.
- 함수의 호출 관계와 데이터 입출력 흐름을 문서화하고, 파일과 함수 위치가 역할에 부합하는지 지속적으로 검토합니다.
- 필요 시 함수 위치를 조정해 블록 규칙을 지키고, 블록 표준화 여부를 평가합니다.
- 어떤 변경이든 시작 전에 현재 문서/구조를 검토하고, 모호함이 있다면 반드시 질문을 통해 문제를 구체화한 뒤 계획을 수립합니다.
- 계획 수립 후에도 의문점이 남으면 다시 질문해 확정하고, 계획을 실행하면서도 필요 시 재질문합니다.
- `docs/architecture.md`를 단일 소스로 유지하고, 구조 결정·변경 사항(폴더 구조, 파일명, 함수명, 순서/패턴)을 즉시 갱신합니다.
- `docs/task.md`에 구조 변경과 관련된 TODO를 기록하고, 완료 시 상태를 업데이트합니다.
- 분석 내용은 `docs/analysis.md`에 정리하고, 새로운 분석이 시작되면 먼저 질문을 통해 목표·데이터·출력물을 명확히 한 뒤 기록합니다.
- 설계 변경을 제안할 때는 대칭화·모듈화·순서화 관점에서 장단점을 평가하고 `docs/architecture.md`에 근거를 명시합니다.

### manager (매니저)
- 임무: 설계 철학을 기준으로 전체 일정과 실행 흐름을 관리하며, 시작 전/중/후에 반드시 질문을 통해 문제를 정의하고 계획을 구체화합니다.
- 작업 절차
  1. **설계 문서 확인**: `docs/architecture.md`를 읽고 변경 사항이 설계 철학(대칭화·모듈화·순서화)에 부합하는지 검토합니다.
  2. **질문으로 요구사항 명확화**: 사용자·설계자에게 질문해 문제 정의, 기대 결과, 제약 조건을 명시합니다.
  3. **할 일 구조화**: `docs/task.md`에 대·중·소 분류(예: `## 대`, `### 중`, `- [ ] 소`)로 계획을 작성해 현재 위치를 한눈에 파악할 수 있게 합니다.
  4. **체크박스 관리**: 진행 상황을 체크박스로 기록하고, 완료 여부·난관·후속 작업을 즉시 업데이트합니다.
  5. **사후 관리**: 완료된 작업은 `docs/review.md` 또는 `docs/history.md`와 연결하고, 필요한 경우 `docs/analysis.md`에 분석 결과를 남깁니다.
- 변경 도중에도 의문이 생기면 즉시 설계자/사용자에게 질문해 계획을 재조정합니다.

### developer (개발자)
- 설계자와 플래너가 합의한 구조와 일정에 맞춰 새로운 함수를 구현합니다.
- 구현 시 함수가 속한 파일과 블록 기준을 준수하며, 입력·출력 계약을 지키는지 확인합니다.
- 기존 블록과의 일관성을 검증하고, 필요한 경우 설계자와 상의해 표준을 업데이트합니다.
- 큰 변경 전에는 반드시 질문을 통해 요구사항을 재확인하고, 작업 계획을 명시한 뒤 실행합니다.
- 구현 결과와 품질 평가는 `docs/review.md`에 기록하고, 개선 사항은 `docs/task.md`로 전파합니다.
- 분석 과정이나 가설 검증은 `docs/analysis.md`에 남겨 후속 작업이 참고할 수 있도록 합니다.
- 실행 중에도 중요한 사건이 발생하면 기록 여부를 설계자/플래너와 질문을 통해 판단합니다.

### evaluator (평가자)
- 목표: 설계 철학(대칭화·모듈화·순서화)과 세부 설계도(`docs/architecture.md`)에 부합하는지, 그리고 요청된 평가 관점별 기준을 만족하는지 검증합니다.
- 평가 전 절차
  1. **질문으로 범위 정의**: 사용자에게 “어느 관점에서 무엇을 어떤 기준으로 평가할지” 반드시 질문해 명확히 합니다.
  2. **설계 문서 확인**: `docs/architecture.md`와 관련 문서를 읽어 의도된 구조/패턴/순서를 이해합니다.
- 평가 수행
  - 관점별 체크리스트를 작성해 `docs/review.md`에 기록합니다. 필요 시 `docs/task.md`에 후속 조치를 등록합니다.
  - 중요한 평가 결과는 `docs/history.md`에 요약하고, 추가 분석이 필요한 경우 `docs/analysis.md`에 상세히 남깁니다.
- 평가 중에도 의문이 생기면 설계자나 매니저에게 질문하여 기준을 재확인합니다.

### documenter (문서화)
- 목표: 모든 이벤트를 기록하지 않고, 프로젝트의 진화를 이해하는 데 필요한 핵심 사건만 육하원칙(When, Where, Who, What, Why, How)에 따라 기록합니다.
- 기록 절차
  - **질문으로 상황 파악**: 사건의 중요도와 맥락을 먼저 질문하여 확인합니다.
  - **docs/history.md 업데이트**: 아래 항목을 채워 기록합니다.
    - When: 타임스탬프(UTC 또는 명시적 로컬 시간)
    - Where: 폴더/파일/함수 등 위치 정보
    - Who: 결정/작업 주체
    - What: 발생한 이벤트, 변경 사항, 문제 등 사실
    - Why: 원인, 의도, 배경
    - How: 해결 방법, 적용된 설정 또는 대응책
- 중복 정보나 사소한 이벤트는 기록하지 않으며, 관련 Summary/Review 문서가 있다면 링크를 추가합니다.
- 기록 후에도 질문이 남으면 설계자나 플래너와 협의하여 필요한 후속 조치를 `docs/task.md` 또는 `docs/review.md`에 연결합니다.
