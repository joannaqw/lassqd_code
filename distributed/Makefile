.PHONY: all clean

all: report.pdf

report.pdf: report.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error report.tex

clean:
	latexmk -c report.tex
