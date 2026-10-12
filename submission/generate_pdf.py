import pathlib, markdown, weasyprint, sys
md_path = pathlib.Path(r'D:\large documents\Large Projects\News_Headline\submission\Telugu_News_Headline_Generation_Final_Report.md')
html = markdown.markdown(md_path.read_text(), extensions=['tables', 'fenced_code', 'toc'])
pdf_path = md_path.with_suffix('.pdf')
weasyprint.HTML(string=html).write_pdf(str(pdf_path))
print('PDF generated at', pdf_path)
