all:
	latex main.tex \
		&& bibtex main.aux \
		&& latex main.tex \
		&& latex main.tex \
		&& dvips -t letter main.dvi \
		&& ps2pdf main.ps
	rm -f *.aux *.bbl *.log *.blg *.dvi *.ps *.out

install:
	mv main.pdf ../

clean:
	rm -f *.aux *.bbl *.log *.blg *.dvi *.ps *.out
