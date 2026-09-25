.PHONY: install prep block featurize train predict validate package run clean

install:
	pip install -r requirements.txt

prep:
	python -m src.pipeline prep

block:
	python -m src.pipeline block

featurize:
	python -m src.pipeline featurize

train:
	python -m src.pipeline train

predict:
	python -m src.pipeline predict

validate:
	python -m src.pipeline validate

package:
	python -m src.pipeline package

run:
	python -m src.pipeline all

clean:
	rm -rf cache output/*.tsv
