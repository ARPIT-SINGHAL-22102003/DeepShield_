from fpdf import FPDF
import hashlib
import datetime
import os

def sha256_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def generate_report(file_path, verdict, confidence, score_v, score_a, alpha_t, heatmap_path):
    pdf = FPDF()
    pdf.add_page()
    
    # ---- HEADER ----
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'DeepShield Forensic Report', ln=True, align='C')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    pdf.ln(5)
    
    # ---- VERDICT ----
# ---- VERDICT ----
    pdf.set_font('Helvetica', 'B', 16)
    if 'FAKE' in verdict:
        pdf.set_text_color(220, 50, 50)   # Red
        clean_verdict = "*** DEEPFAKE DETECTED ***"
    else:
        pdf.set_text_color(50, 180, 50)   # Green
        clean_verdict = "*** LIKELY REAL ***"
    pdf.cell(0, 12, clean_verdict, ln=True, align='C')
    
    # ---- DETAILS ----
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Detection Details', ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, f'Confidence Score : {confidence:.1%}', ln=True)
    
    if score_v is not None:
        pdf.cell(0, 7, f'Visual Score     : {score_v:.4f}', ln=True)
    else:
        pdf.cell(0, 7, f'Visual Score     : N/A', ln=True)
        
    if score_a is not None:
        pdf.cell(0, 7, f'Audio Score      : {score_a:.4f}', ln=True)
    else:
        pdf.cell(0, 7, f'Audio Score      : N/A', ln=True)
        
    pdf.cell(0, 7, f'Fusion Weight    : {alpha_t:.4f}', ln=True)
    pdf.ln(5)
    
    # ---- CHAIN OF CUSTODY ----
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Chain of Custody', ln=True)
    pdf.set_font('Courier', '', 8)
    pdf.cell(0, 6, f'File    : {os.path.basename(file_path)}', ln=True)
    pdf.cell(0, 6, f'SHA-256 : {sha256_hash(file_path)}', ln=True)
    pdf.ln(5)
    
    # ---- HEATMAP IMAGE ----
    if heatmap_path and os.path.exists(heatmap_path):
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Grad-CAM Forensic Heatmap', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, 'Red = Fake regions | Orange = Suspicious | Blue = Real', ln=True)
        pdf.ln(3)
        pdf.image(heatmap_path, w=150)
    
    # ---- SAVE ----
    report_path = file_path + '_report.pdf'
    pdf.output(report_path)
    return report_path