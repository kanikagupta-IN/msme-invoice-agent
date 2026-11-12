import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_invoice_pdf(filename, vendor, date, inv_num, items, delivery_fee, notes=""):
    """
    Generates a clean, structured business invoice PDF for local testing.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=24, leading=28, textColor=colors.HexColor("#1A365D"))
    meta_style = ParagraphStyle('MetaText', parent=styles['Normal'], fontSize=10, leading=14)
    item_style = ParagraphStyle('ItemText', parent=styles['Normal'], fontSize=10, leading=12)
    notes_style = ParagraphStyle('NotesText', parent=styles['Italic'], fontSize=9, leading=13, textColor=colors.HexColor("#4A5568"))

    # 1. Header Section
    story.append(Paragraph(f"<b>{vendor}</b>", title_style))
    story.append(Paragraph("Official Wholesale Supplier Invoice", meta_style))
    story.append(Spacer(1, 15))
    
    # 2. Metadata Block Matrix
    meta_data = [
        [Paragraph(f"<b>Invoice Number:</b> {inv_num}", meta_style), Paragraph(f"<b>Billing Date:</b> {date}", meta_style)],
        [Paragraph("<b>Bill To:</b> MSME Retail Shop Workspace", meta_style), Paragraph("<b>Terms:</b> Net 30 Days", meta_style)]
    ]
    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(meta_table)
    story.append(Spacer(1, 20))
    
    # 3. Line Items Table Compilation
    table_data = [[Paragraph("<b>Description</b>", item_style), Paragraph("<b>Qty</b>", item_style), Paragraph("<b>Unit Price (₹)</b>", item_style), Paragraph("<b>Total (₹)</b>", item_style)]]
    
    subtotal = 0.0
    for desc, qty, price in items:
        item_total = qty * price
        subtotal += item_total
        table_data.append([
            Paragraph(desc, item_style),
            Paragraph(str(qty), item_style),
            Paragraph(f"{price:.2f}", item_style),
            Paragraph(f"{item_total:.2f}", item_style)
        ])
        
    # Append Financial Summary Blocks
    table_data.append([Paragraph("", item_style), Paragraph("", item_style), Paragraph("<b>Subtotal:</b>", item_style), Paragraph(f"{subtotal:.2f}", item_style)])
    table_data.append([Paragraph("", item_style), Paragraph("", item_style), Paragraph("<b>Delivery Charges:</b>", item_style), Paragraph(f"{delivery_fee:.2f}", item_style)])
    
    grand_total = subtotal + delivery_fee
    table_data.append([Paragraph("", item_style), Paragraph("", item_style), Paragraph("<b>Grand Total:</b>", item_style), Paragraph(f"<b>{grand_total:.2f}</b>", item_style)])
    
    # Apply Formatting to Items Matrix Table
    items_table = Table(table_data, colWidths=[240, 50, 110, 120])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-4), 0.5, colors.HexColor("#CBD5E0")),
        ('LINEBELOW', (2,-3), (3,-1), 1, colors.HexColor("#1A365D")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (1,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    
    story.append(items_table)
    story.append(Spacer(1, 20))
    
    # 4. Optional Footnotes / Fine-Print Additions (Crucial for Vector RAG Testing)
    if notes:
        story.append(Paragraph("<b>Important Notice / Policy Notes:</b>", item_style))
        story.append(Spacer(1, 5))
        story.append(Paragraph(notes, notes_style))
        
    doc.build(story)
    print(f"Generated mock document: {filename}")

if __name__ == "__main__":
    # Ensure local directory structure exists
    os.makedirs("data/sample_invoices", exist_ok=True)
    
    print("Generating test dataset documents...")
    
    # --- SCENARIO 1: ABC WHOLESALERS (Delivery Fee Jump) ---
    # Month A (August) -> Delivery Fee is ₹450
    create_invoice_pdf(
        filename="data/sample_invoices/abc_wholesalers_aug.pdf",
        vendor="ABC Wholesalers",
        date="2026-08-10",
        inv_num="INV-2026-089",
        items=[("Bulk Cardboard Retail Boxes", 100, 12.0), ("Heavy Duty Packaging Tape", 10, 85.0)],
        delivery_fee=450.0,
        notes="Standard shipping baseline logistics applied to regional distribution nodes."
    )
    
    # Month B (September) -> Delivery Fee spikes to ₹600 with an explicit structural explanation footnote
    create_invoice_pdf(
        filename="data/sample_invoices/abc_wholesalers_sep.pdf",
        vendor="ABC Wholesalers",
        date="2026-09-12",
        inv_num="INV-2026-142",
        items=[("Bulk Cardboard Retail Boxes", 120, 12.0), ("Heavy Duty Packaging Tape", 15, 85.0)],
        delivery_fee=600.0,
        notes="Please note: Shipping charges have been revised upwards to ₹600 this month due to global diesel fuel surcharges and carrier driver resource shortages."
    )

    # --- SCENARIO 2: XYZ MANUFACTURING (Flat Delivery Fees) ---
    create_invoice_pdf(
        filename="data/sample_invoices/xyz_manufacturing_sep.pdf",
        vendor="XYZ Manufacturing Ltd",
        date="2026-09-15",
        inv_num="B-99823",
        items=[("Premium Matte Finished Plastics", 50, 240.0)],
        delivery_fee=350.0,
        notes="Fixed rate shipping contract active until early next year."
    )
    
    print("\nDone! Three mock invoices are ready in 'data/sample_invoices/'.")
