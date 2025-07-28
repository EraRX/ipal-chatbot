import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def genereer_pdf(tekst: str) -> bytes:
    """
    Zet een tekst om naar een eenvoudige PDF (A4) en geeft de bytes terug.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    text_obj = c.beginText(40, height - 50)
    text_obj.setFont("Helvetica", 12)
    for line in tekst.split("\n"):
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
