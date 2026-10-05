from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
import re
src=Path('report/report_source.md').read_text()
out=Path('report/quantitative_portfolio_report.pdf'); out.parent.mkdir(exist_ok=True)
doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=2*cm,leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm,title='Portfolio Optimization Under Uncertainty')
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='CoverTitle',parent=styles['Title'],fontSize=24,leading=30,alignment=TA_CENTER,spaceAfter=20,textColor=colors.HexColor('#17365D'))); styles.add(ParagraphStyle(name='H1x',parent=styles['Heading1'],fontSize=15,leading=19,spaceBefore=12,spaceAfter=8,textColor=colors.HexColor('#17365D'))); styles.add(ParagraphStyle(name='H2x',parent=styles['Heading2'],fontSize=11,leading=14,spaceBefore=9,spaceAfter=5)); styles.add(ParagraphStyle(name='Bodyx',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=7)); styles.add(ParagraphStyle(name='Smallx',parent=styles['BodyText'],fontSize=8,leading=10,spaceAfter=5))
def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
story=[Spacer(1,4*cm),Paragraph('Portfolio Optimization<br/>Under Uncertainty',styles['CoverTitle']),Paragraph('A comparative study of Modern Portfolio Theory, risk parity, robust estimation and out-of-sample performance',styles['Heading2']),Spacer(1,1*cm),Paragraph('Quantitative Finance Research Project',styles['Bodyx']),Paragraph('Methodology report | Version 0.1 | 5 October 2026',styles['Bodyx']),Spacer(1,2*cm),Paragraph('<b>Important status note</b>: this is a reproducible research-design report. It contains no fabricated market data or performance findings. The empirical results must be generated and validated by running the pipeline against retrieved data.',styles['Bodyx']),PageBreak()]
section_count=0
for line in src.splitlines():
 s=line.strip()
 if not s: story.append(Spacer(1,3)); continue
 if s.startswith('# '):
  if s.startswith('# Portfolio'): continue
  if section_count: story.append(PageBreak())
  section_count += 1
  story.append(Paragraph(esc(s[2:]),styles['H1x']))
 elif s.startswith('## '):
  if section_count: story.append(PageBreak())
  section_count += 1
  story.append(Paragraph(esc(s[3:]),styles['H1x']))
 elif s.startswith('### '):
  if section_count: story.append(PageBreak())
  section_count += 1
  story.append(Paragraph(esc(s[4:]),styles['H1x']))
 elif s.startswith('- '): story.append(Paragraph('• '+esc(s[2:]),styles['Bodyx']))
 else:
  s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s); s=re.sub(r'\*(.*?)\*',r'<i>\1</i>',s); s=re.sub(r'`(.*?)`',r'<font name="Courier">\1</font>',s)
  s=s.replace('\\(', '').replace('\\)', '').replace('\\[','').replace('\\]','')
  story.append(Paragraph(s,styles['Bodyx']))
def footer(canvas,doc):
 canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.grey); canvas.drawString(2*cm,1.1*cm,'Portfolio Optimization Under Uncertainty | Research design, not investment advice'); canvas.drawRightString(A4[0]-2*cm,1.1*cm,str(doc.page)); canvas.restoreState()
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(out, out.stat().st_size)
