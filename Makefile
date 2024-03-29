.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

all: build

git: docs avcv
	git add -A && git commit -v && git push

build: 
	nbdev_build_lib && pip install -e ./

avcv: $(SRC)
	nbdev_build_lib
	touch avcv

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs:
	nbdev_clean_nbs && nbdev_build_docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

lib:
	nbdev_build_lib && pip install -e ./
