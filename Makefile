# 项目工具 — 仅提供各 predictor 环境的 Python 路径
PYTHON_SUNDIAL = src/models/predictors/Sundial/.venv/bin/python
PYTHON_TIMESFM = src/models/predictors/TimesFM25/.venv/bin/python

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: sundial
sundial:  ## 用 Sundial 环境运行: make sundial ARGS="main.py --imputator.model_type SAITS"
	$(PYTHON_SUNDIAL) $(ARGS)

.PHONY: timesfm
timesfm:  ## 用 TimesFM 环境运行: make timesfm ARGS="main.py --predictor.model_type TimesFM25"
	$(PYTHON_TIMESFM) $(ARGS)

.PHONY: python
python: sundial  ## 默认使用 Sundial 环境 (同 make sundial)
