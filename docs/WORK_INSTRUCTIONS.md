# �۾� ��ħ��: ���� ���� ����Ʈ �� ���� �ܰ� ����

�� ������ �ٸ� �۾��ڰ� ���� ������ ������ �� �ֵ��� ��Ȳ�� �ٰ�, �׸��� ���� �ܰ� ������ ������ ��ħ���Դϴ�.

## 1. ���� �ٰ� ��� (�ڵ� ��� Ȯ�� ����)

1) Web -> Pipeline ����
- `src/web/app.py`�� `src.pipeline.InspectionPipeline`�� ���� �����.
- `/inspect`, `/inspect_v2`, `/batch`, `/recompute` ��� `InspectionPipeline` ����.

2) Web -> v7 ���� ȣ��
- `/inspect` ���ο��� `src.web.routers.v7.inspect`�� ���� ȣ���� v7 ���� �帧�� ������.

3) engine_v7 vs engine_v7
- `src/engine_v7/__init__.py`�� `sys.modules` alias�� ���� `src.engine_v7.core`�� �����.
- `src/engine_v7`�� �� ������ ����.

4) pipeline�� ���� ������
- `src/pipeline.py`�� `src.engine_v7.core.*`�� ���� import (alias ���� impl�� ����).
- v7 ��/���� ����(������Ʈ��, analyzer ��)�� pipeline ���ο��� ���� ȣ��.

5) ���� ��� �̿�ȭ
- SKU ����: ��Ʈ `config/sku_db`�� `load_sku_config()`�� ���� (`src/web/app.py`).
- ���� ����: `src/config/v7_paths.py` �� `src/engine_v7/configs` ��� (`src/pipeline.py`).

�� �������� ���� ������ ȥ��(���� ���� ����, web ���� ��� ȥ��, ���� �л�)�� Ȯ�ε�.

## 2. ���� ��� (Critical Findings)

- ���� ���� ����: `engine_v7`�� alias �������̸�, ��ü�� `engine_v7`.
- ���������� ��ġ: `src/pipeline.py`�� ��Ʈ�� ����ǰ� ���� ������ �����յ�.
- ���� ��� ȥ��: Web ������ pipeline ������ v7 direct ȣ���� ȥ��.
- ���� ���� �л�: ��Ʈ config/�� ���� configs/�� ���� ������ �Ҹ�Ȯ.

## 3. ���� ���� (Recommendations)

### Phase 1: ���� API ���̾� ����
��ǥ: Web/Pipeline�� ���� ���� ��� ���� API�� ����ϵ��� ����.

- `src/engine_v7/api.py` (�Ǵ� `src/engine/api.py`) �ż�
- API ����:
  - `inspect_single(image_bgr, sku, ink, cfg_override=None) -> DecisionPayload`
  - `register_std(image_bgrs, sku, ink, cfg_override=None) -> RegistrationResult`
  - `load_models(sku, ink, cfg_hash=None) -> ModelBundle`
- Web �� pipeline�� �� API�� ȣ���ϵ��� �����͸�

### Phase 2: ������ ����
��ǥ: Web �������� ���� ���� ��� ���� import ����.

- `src/web/routers/v7.py`���� `src.engine_v7.core.*` ���� import ����
- `src/web/app.py`�� v7 ���� ȣ���� ���� API ȣ��� ��ü
- pipeline�� v7 ȣ�⵵ ������ API�� ����

### Phase 3: ���� ���� (alias ����)
��ǥ: engine_v7 -> engine_v7���� ��ü �̵� �� alias ����.

- `src/engine_v7`�� `src/engine_v7`�� ����
- `src/engine_v7/__init__.py`�� alias ���� ����
- ��� import ��θ� `src.engine_v7.*`�� ����

### Phase 4: ���� ���� �Ͽ�ȭ
��ǥ: config �л� �ؼ� �� �δ� ����.

- `config/` (SKU � ����) vs `src/engine_v7/configs/` (���� �⺻��) ���� ����
- `ConfigLoader`�� �ϳ��� �����Ͽ� �ε� ��� �ϰ�ȭ

## 4. ���� ��ȹ (Action Items)

1) ���� API ���̾� ����
- ���� �Լ� ��� ����
- �Է�/��� ��Ű�� Ȯ��

2) Web/Pipeline �����͸�
- v7 API ȣ��� ����
- ���� ���� ��� direct import ����

3) ���� ���͸� �̵�
- `engine_v7` -> `engine_v7`�� �̵�
- alias ����
- import ��� ���� ����

4) ���� �ε� ����
- ConfigLoader ����
- ��Ʈ config�� ���� configs ���� ����ȭ

## 5. �۾� ���� ���� (������ ���� ����)

1) API ���̾� ���� (�� ���� �߰�)
2) Web/Pipeline�� API�� ����ϵ��� �ܰ��� ����
3) alias ���� �� ���͸� �̵�
4) ���� �ε� ���� �� ����ȭ

## 6. ���ǻ���

- alias ���� ������ import ��� ������ API ���̾� ������ ����Ǿ�� ������.
- `src/web/routers/v7.py`�� ���� ���ο� ����� �����ϰ� �־� ���� ������ ŭ.
- �۾� ���ķ� �׽�Ʈ(Ư�� API/���� �׽�Ʈ) ���� ����.

## 7. �� �۾� üũ����Ʈ

### A. �غ� �ܰ�
- [ ] ���� ����/������ ����ȭ (engine_v7, engine_v7, pipeline, web)
- [ ] ���� ���� ���� (API ���̾� ����, ���͸� �̵� ����)
- [ ] �⺻ �׽�Ʈ �ó����� ���� (inspect, inspect_v2, batch, v7 endpoints)

### B. ���� API ���̾� ����
- [ ] `src/engine_v7/api.py` �ż� (�Ǵ� `src/engine/api.py` Ȯ��)
- [ ] ���� �Լ� �ñ״�ó ����
  - [ ] `inspect_single(...)`
  - [ ] `register_std(...)`
  - [ ] `load_models(...)`
- [ ] ���� ��� ȣ���� API ���η� ĸ��ȭ
- [ ] API ����� ��Ű��/���� �ڵ鸵 �Ծ� ����

### C. Web/Pipeline ������ ����
- [ ] `src/web/app.py`���� v7 direct ȣ���� API ȣ��� ����
- [ ] `src/web/routers/v7.py`�� `src.engine_v7.core.*` ���� import ����
- [ ] `src/pipeline.py`�� v7 ȣ���� API�� ����
- [ ] Web ������ `engine_v7.api`�� �ٶ󺸵��� ����

### D. ���� ���͸� ����
- [ ] `src/engine_v7` -> `src/engine_v7`�� �̵�
- [ ] `src/engine_v7/__init__.py` alias ����
- [ ] import ��� ���� ���� (`engine_v7` ����)

### E. ���� �ε� ����
- [ ] `config/` vs `src/engine_v7/configs/` ���� ���� ����ȭ
- [ ] ConfigLoader ����/����
- [ ] SKU config �ε� ��� ���

### F. ���� �ܰ�
- [ ] /inspect, /inspect_v2, /batch, /recompute ��� ����
- [ ] v7 ���� API ���� ���� Ȯ��
- [ ] ���� ��� ���� ȣȯ�� ����
- [ ] �ּ� ȸ�� �׽�Ʈ ����

### G. ����/����
- [ ] ����� import ��Ģ ����ȭ
- [ ] ���� API ��� ���̵� �߰�
- [ ] ���� ���� Ʈ�� ������Ʈ
