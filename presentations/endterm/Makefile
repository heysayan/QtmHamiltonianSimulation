# Makefile for LaTeX Beamer Presentation

# Main file
MAIN = presentation

# LaTeX compiler
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode

# Output
PDF = $(MAIN).pdf

# Auxiliary files
AUX_FILES = *.aux *.log *.nav *.out *.snm *.toc *.synctex.gz *.fls *.fdb_latexmk

.PHONY: all clean view help

# Default target
all: $(PDF)

# Compile PDF (run twice for references)
$(PDF): $(MAIN).tex
	@echo "Compiling LaTeX presentation..."
	@$(LATEX) $(LATEX_FLAGS) $(MAIN).tex > /dev/null
	@echo "Second pass for references..."
	@$(LATEX) $(LATEX_FLAGS) $(MAIN).tex > /dev/null
	@echo "✓ Success! Created $(PDF)"

# Open PDF viewer
view: $(PDF)
	@if command -v evince > /dev/null; then \
		evince $(PDF) & \
	elif command -v okular > /dev/null; then \
		okular $(PDF) & \
	elif command -v xdg-open > /dev/null; then \
		xdg-open $(PDF) & \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -f $(AUX_FILES)
	@echo "✓ Cleaned"

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	@rm -f $(PDF)
	@echo "✓ All clean"

# Show help
help:
	@echo "LaTeX Beamer Presentation Makefile"
	@echo "===================================="
	@echo ""
	@echo "Targets:"
	@echo "  make          - Compile presentation.pdf"
	@echo "  make view     - Compile and open in PDF viewer"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean- Remove all generated files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (install texlive-full on Ubuntu/Debian)"
	@echo "  - PNG diagram files in same directory"
