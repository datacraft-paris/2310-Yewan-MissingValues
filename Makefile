SHELL := /bin/bash -i # Bourne-Again SHell command-line interpreter on Linux.
PYTHON := python3 # Python interpreter.
### 

# Hack for displaying help message in Makefile
help: 
	@grep -E '(^[0-9a-zA-Z_-]+:.*?##.*$$)' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}'

###

# This file is used for bypass variables and set some specific env. vars.
.env.bash:
	touch .env.bash

venv-build: ## Create Python environement with venv.
venv-build: venv/main/bin/activate
venv/main/bin/activate: .env.bash
	-( \
	. .env.bash \
	&& test -d venv/main \
	|| $(PYTHON) -m venv venv/main \
	)
	-( \
	. .env.bash \
	&& . venv/main/bin/activate \
	&& pip install -U pip \
	&& pip install -r requirements.txt \
	)

venv-start-lab: ## Start jupyter lab with under venv.
venv-start-lab: .env.bash venv/main/bin/activate
	. .env.bash \
	&& . venv/main/bin/activate \
	&& jupyter lab --no-browser

# Because sometime Jupyter lab freeze when performing visualization.
venv-start-nb: ## Start jupyter notebook with venv.
venv-start-nb: .env.bash venv/main/bin/activate
	. .env.bash \
	&& . venv/main/bin/activate \
	&& jupyter notebook --no-browser

nbs-clear-output: ## Clear all notebooks.
nbs-clear-output:
	@for i in *.ipynb;do \
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --clear-output --inplace $$i; \
	done

data-unzip: ## Unzip the data (script/unzip.bash).
data-unzip: script/unzip.bash data.zip 
	(./$<)

data-zip: ## Zip the data (script/zip.bash).
data-zip: script/zip.bash data.zip 
	(./$<)

clean: ## Cleaning all files and directories generated.
clean:
	-(rm -rf __pycache__)
	-(rm -rf __MACOSX)
	-(rm -rf venv)
	-(rm -rf data)


