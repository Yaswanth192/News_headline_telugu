import pathlib
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

md_path = pathlib.Path(r'D:\large documents\Large Projects\News_Headline\submission\Telugu_News_Headline_Generation_Final_Report.md')
pdf_path = md_path.with_suffix('.pdf')

styles = getSampleStyleSheet()
# Use existing style names; no redefinition needed
heading1 = styles['Heading1']
heading2 = styles['Heading2']
normal = styles['Normal']

story = []

with md_path.open(encoding='utf-8') as f:
    lines = [ln.rstrip('\n') for ln in f]

i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('# '):
        story.append(Paragraph(line[2:], heading1))
        story.append(Spacer(1, 12))
        i += 1
    elif line.startswith('## '):
        story.append(Paragraph(line[3:], heading2))
        story.append(Spacer(1, 8))
        i += 1
    elif line.startswith('|') and '|' in line[1:]:
        # collect contiguous table rows
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            cells = [c.strip() for c in lines[i].strip('|').split('|')]
            rows.append(cells)
            i += 1
        tbl = Table(rows, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica')
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))
    elif line.strip() == '':
        i += 1
    else:
        # accumulate paragraph lines until a blank line or a heading/table marker
        para_lines = []
        while i < len(lines) and lines[i].strip() != '' and not lines[i].startswith('#') and not lines[i].startswith('##') and not (lines[i].startswith('|') and '|' in lines[i][1:]):
            para_lines.append(lines[i].strip())
            i += 1
        story.append(Paragraph(' '.join(para_lines), normal))
        story.append(Spacer(1, 6))

# Build PDF
doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
doc.build(story)
print('PDF generated at', pdf_path)
