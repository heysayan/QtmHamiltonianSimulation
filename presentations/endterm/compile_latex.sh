#!/bin/bash
# Script to compile LaTeX Beamer presentation

echo "Compiling LaTeX Beamer presentation..."
echo "======================================"

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null
then
    echo "ERROR: pdflatex not found!"
    echo "Please install LaTeX (texlive-full on Ubuntu/Debian)"
    echo "  sudo apt-get install texlive-full"
    exit 1
fi

# Compile the presentation (run twice for references)
echo "First pass..."
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1

echo "Second pass (for references)..."
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f presentation.aux presentation.log presentation.nav presentation.out presentation.snm presentation.toc

if [ -f "presentation.pdf" ]; then
    echo ""
    echo "✓ Success! PDF created: presentation.pdf"
    echo ""
    echo "File size: $(du -h presentation.pdf | cut -f1)"
    echo ""
    echo "To view: evince presentation.pdf  (or your preferred PDF viewer)"
else
    echo ""
    echo "✗ Error: PDF was not created"
    echo "Check presentation.log for errors"
fi
