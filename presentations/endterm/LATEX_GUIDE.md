# LaTeX Beamer Presentation Guide

## Quick Start

### Option 1: Use the Compile Script (Easiest)

```bash
cd /path/to/presentations/endterm
./compile_latex.sh
```

This will generate `presentation.pdf` automatically!

### Option 2: Manual Compilation

```bash
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for references
```

### Option 3: Using Latexmk (Recommended for Auto-compilation)

```bash
latexmk -pdf -pvc presentation.tex
```

The `-pvc` flag enables continuous preview with auto-recompilation on file changes.

## Prerequisites

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

### macOS
```bash
brew install --cask mactex
```

### Windows
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

## Presentation Features

### 🎨 Professional Design
- **Theme**: Madrid with custom colors
- **Aspect Ratio**: 16:9 (widescreen)
- **Colors**: Dark blue theme with custom accents
- **Fonts**: Clean, readable fonts for presentations

### 📊 Content Structure

1. **Introduction** (4 slides)
   - Problem motivation
   - Algorithm timeline
   - Overview

2. **Trotterization** (4 slides)
   - Core concept
   - Visual diagram
   - Complexity analysis

3. **Taylor-LCU** (3 slides)
   - Taylor series approach
   - Block encoding
   - PREPARE-SELECT-PREPARE† structure

4. **QSP** (1 slide)
   - Quantum Signal Processing
   - Polynomial transformations

5. **Qubitization** (1 slide)
   - Optimal quantum walks
   - Eigenvalue encoding

6. **QSVT** (3 slides)
   - Grand unification
   - Framework diagram
   - Properties and advantages

7. **Benchmarks** (3 slides)
   - Complexity comparison
   - Radar chart
   - Detailed results table

8. **Grover Application** (3 slides)
   - Hamiltonian formulation
   - Visual explanation
   - Comparison results

9. **Conclusions** (4 slides)
   - Algorithm summary
   - Practical considerations
   - Key takeaways
   - Future directions

10. **Closing** (2 slides)
    - Implementation details
    - References
    - Q&A

**Total: ~30 slides** (perfect for 45-60 minute presentation)

### 🖼️ Included Diagrams

All 8 PNG diagrams are automatically included:
- `algorithm_timeline.png`
- `complexity_comparison.png`
- `algorithm_radar.png`
- `trotter_diagram.png`
- `lcu_diagram.png`
- `qsvt_diagram.png`
- `grover_hamiltonian_diagram.png`
- `benchmark_results.png`

### 📐 Mathematical Content

Properly formatted equations using LaTeX:
- Complex exponentials: $e^{-iHt}$
- Product formulas
- Taylor series expansions
- Complexity bounds (Big-O notation)
- Quantum states and operators

### 📋 Tables

Professional tables using `booktabs`:
- Algorithm complexity comparison
- Benchmark results
- Grover implementation comparison

## Customization

### Change Theme
```latex
\usetheme{Madrid}        % Options: Madrid, Berlin, Copenhagen, etc.
\usecolortheme{default}  % Options: default, crane, dove, etc.
```

### Change Colors
```latex
\definecolor{darkblue}{RGB}{0,51,102}    % Title color
\definecolor{lightblue}{RGB}{52,152,219} % Accent color
```

### Change Aspect Ratio
```latex
\documentclass[aspectratio=169]{beamer}  % 16:9 widescreen
% Options: 43 (4:3), 169 (16:9), 1610 (16:10)
```

### Modify Title
```latex
\title[Short Title]{Full Title}
\subtitle{Your Subtitle}
\author{Your Name}
\institute{Your Institution}
\date{\today}
```

## Advanced Features

### Adding Animations

Use `\pause` to reveal content step-by-step:
```latex
\begin{frame}{Example}
First item
\pause
Second item (appears on click)
\pause
Third item (appears on another click)
\end{frame}
```

### Using Blocks

Different block types:
```latex
\begin{block}{Title}
Regular block
\end{block}

\begin{alertblock}{Warning}
Alert block (red)
\end{alertblock}

\begin{exampleblock}{Example}
Example block (green)
\end{exampleblock}
```

### Two-Column Layout

```latex
\begin{columns}
\column{0.5\textwidth}
Left column content

\column{0.5\textwidth}
Right column content
\end{columns}
```

### Adding Code

```latex
\begin{verbatim}
def quantum_algorithm():
    return "Hello Quantum World"
\end{verbatim}
```

## Presentation Tips

### Before Presenting

1. **Test the PDF**: Open and scroll through all slides
2. **Check Images**: Ensure all diagrams display correctly
3. **Test on Projector**: Verify colors and readability
4. **Practice Timing**: Aim for ~2 minutes per slide
5. **Prepare Notes**: Use separate notes document if needed

### During Presentation

- **Navigation**:
  - Space/Arrow Right: Next slide
  - Arrow Left: Previous slide
  - Home: First slide
  - End: Last slide

- **Presenter Mode** (if supported):
  - Shows current slide, next slide, and timer
  - Available in Adobe Reader, Okular, etc.

### Recommended Settings

For **Adobe Acrobat Reader**:
- View → Full Screen Mode
- Edit → Preferences → Full Screen → Set transitions

For **Okular** (Linux):
- View → Presentation
- Use presenter screen if available

For **Preview** (macOS):
- View → Slideshow
- Use presenter display

## Troubleshooting

### PDF Not Created

1. Check for LaTeX errors:
```bash
pdflatex presentation.tex
# Look for ERROR messages
```

2. Common issues:
   - Missing packages: Install `texlive-full`
   - Image not found: Verify PNG files are in same directory
   - Syntax errors: Check LaTeX syntax

### Images Not Showing

```bash
# Verify images exist
ls -lh *.png

# Ensure LaTeX can find them
# Images should be in same directory as .tex file
```

### Compilation Warnings

Most warnings are safe to ignore. Critical ones start with `ERROR`.

### Font Issues

If fonts look wrong:
```bash
# Update font cache
sudo fc-cache -fv
```

## Output Files

After compilation, you'll have:

- **`presentation.pdf`** ← Main output (this is what you want!)
- `presentation.aux` - Auxiliary file (can delete)
- `presentation.log` - Compilation log (useful for debugging)
- `presentation.nav` - Navigation helper (can delete)
- `presentation.out` - Hyperref output (can delete)
- `presentation.snm` - Beamer navigation (can delete)
- `presentation.toc` - Table of contents (can delete)

The compile script automatically cleans up auxiliary files.

## Version Control

### .gitignore for LaTeX

Already included, but for reference:
```gitignore
*.aux
*.log
*.nav
*.out
*.snm
*.toc
*.synctex.gz
*.fls
*.fdb_latexmk
```

Keep:
- `presentation.tex` (source)
- `presentation.pdf` (output)
- `*.png` (diagrams)

## Converting to Other Formats

### LaTeX → PowerPoint

Use `beamer2ppt` or export to images:
```bash
# Convert each slide to image
pdftoppm presentation.pdf slide -png

# Import images into PowerPoint
```

### LaTeX → HTML

Use `tex4ht`:
```bash
htlatex presentation.tex
```

### LaTeX → Google Slides

1. Generate PDF
2. Upload to Google Drive
3. Open with Google Slides

## Tips for Different Audiences

### For Technical Audience
- Include mathematical details
- Show complexity proofs
- Discuss implementation challenges
- Keep all slides

### For General Audience
- Skip mathematical derivations
- Focus on visualizations
- Emphasize applications
- Consider removing slides 10-15 (detailed math)

### For Research Group
- Add extra slides on:
  - Open problems
  - Recent developments
  - Comparison with literature
  - Implementation details

## Creating Handouts

Generate handout version with multiple slides per page:

```latex
\documentclass[handout]{beamer}  % Add 'handout' option
```

Then compile:
```bash
pdflatex presentation.tex
pdfnup presentation.pdf --nup 2x3 --outfile handout.pdf
```

This creates 6 slides per page for printing.

## Additional Resources

### Beamer Documentation
- [Beamer User Guide](https://ctan.org/pkg/beamer)
- [Overleaf Beamer Tutorial](https://www.overleaf.com/learn/latex/Beamer)

### Themes Gallery
- [Beamer Theme Matrix](https://hartwork.org/beamer-theme-matrix/)
- Browse and choose themes visually

### LaTeX Math
- [LaTeX Math Symbols](https://www.ctan.org/pkg/comprehensive)
- [Detexify](http://detexify.kirelabs.org/classify.html) - Draw symbol to find LaTeX command

## Quick Reference Card

### Essential Commands
| Command | Action |
|---------|--------|
| `\section{Title}` | Create new section |
| `\begin{frame}` | Start new slide |
| `\frametitle{Title}` | Set slide title |
| `\pause` | Add animation break |
| `\includegraphics[width=0.8\textwidth]{file.png}` | Insert image |
| `\begin{itemize}` | Bullet list |
| `\begin{enumerate}` | Numbered list |
| `\begin{block}{Title}` | Create block |

### Math Mode
| Code | Output |
|------|--------|
| `$x^2$` | Inline math |
| `\[ x^2 \]` | Display math |
| `\begin{equation}` | Numbered equation |
| `\frac{a}{b}` | Fraction |
| `\sum_{i=1}^n` | Summation |

## Conclusion

Your LaTeX Beamer presentation is ready to compile and present! 🎉

**Quick Commands:**
```bash
# Compile
./compile_latex.sh

# View
evince presentation.pdf

# Present
# Open in full-screen mode with your PDF viewer
```

**Need help?** Check `presentation.log` for compilation errors or consult the Beamer documentation.

---

Happy Presenting! 🚀
