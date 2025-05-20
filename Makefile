.PHONY: quality style

check_dirs := ./data_process ./eval ./__assets__

exclude_dirs := groundingsam2,lama_with_maskdino,third_party

quality:
	ruff check $(check_dirs) --exclude $(exclude_dirs)
	ruff format --check $(check_dirs) --exclude $(exclude_dirs)

style:
	ruff check $(check_dirs) --fix --exclude $(exclude_dirs)
	ruff format $(check_dirs) --exclude $(exclude_dirs)
