"""Throwaway generator for synthetic demo documents (PRD section 9).

Clean fonts, no glare/skew -> OCR-friendly, reliable for the live demo.
All data is fake. Run:  python sample_docs/generate_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = Path(r"C:\Windows\Fonts")
A4 = (1654, 2339)  # A4 @ ~140 DPI

FONT = str(FONTS / "arial.ttf")
FONT_BOLD = str(FONTS / "arialbd.ttf")
# Consolas digits OCR badly (zeros -> 'Q'); Arial Bold digits are reliable.
FONT_MONO = str(FONTS / "arialbd.ttf")


def _font(spec: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(spec, size)


def _text(draw, xy, s, spec=FONT, size=28, fill=(20, 20, 20), anchor=None):
    draw.text(xy, s, font=_font(spec, size), fill=fill, anchor=anchor)


def make_bank_statement(out: Path) -> Path:
    img = Image.new("RGB", A4, (255, 255, 255))
    d = ImageDraw.Draw(img)
    _text(d, (80, 60), "HDFC BANK LTD", FONT_BOLD, 40)
    _text(d, (80, 120), "Statement of Account", FONT, 30)
    _text(d, (1100, 60), "Statement Period: 01-Apr-2026 to 30-Jun-2026", FONT, 24)
    d.line((80, 175, 1574, 175), fill=(0, 0, 0), width=3)

    _text(d, (80, 210), "Account Holder: ARJUN MEHTA", FONT_BOLD, 30)
    _text(d, (80, 260), "Account Number: 5010023456789012", FONT_MONO, 30)
    _text(d, (80, 310), "IFSC Code: HDFC0001234", FONT_MONO, 28)
    _text(d, (80, 360), "Branch: Koramangala, Bengaluru 560034", FONT, 26)
    _text(d, (80, 410), "Registered Mobile: 9876543210", FONT, 26)
    _text(d, (80, 460), "Email: arjun.mehta@example.com", FONT, 26)

    y = 560
    _text(d, (80, y), "Date", FONT_BOLD, 26)
    _text(d, (330, y), "Description", FONT_BOLD, 26)
    _text(d, (1150, y), "Amount (INR)", FONT_BOLD, 26)
    d.line((80, y + 45, 1574, y + 45), fill=(0, 0, 0), width=2)
    rows = [
        ("02-Apr-2026", "SALARY CREDIT - INFOTECH SOLUTIONS PVT LTD", "85,000.00"),
        ("05-Apr-2026", "UPI/REF/982134/Rent Payment", "24,500.00"),
        ("15-Apr-2026", "Card Purchase BIGBASKET", "3,214.00"),
        ("01-May-2026", "SALARY CREDIT - INFOTECH SOLUTIONS PVT LTD", "85,000.00"),
        ("12-May-2026", "NEFT/UTILITIES/BESCOM", "2,180.00"),
        ("01-Jun-2026", "SALARY CREDIT - INFOTECH SOLUTIONS PVT LTD", "85,000.00"),
        ("20-Jun-2026", "Card Purchase SWIGGY", "640.00"),
    ]
    y += 65
    for date, desc, amt in rows:
        _text(d, (80, y), date, FONT, 24)
        _text(d, (330, y), desc, FONT, 24)
        _text(d, (1150, y), amt, FONT_MONO, 24)
        y += 55
    d.line((80, y + 10, 1574, y + 10), fill=(0, 0, 0), width=2)
    _text(d, (80, y + 40), "Closing Balance: INR 3,42,118.00", FONT_BOLD, 28)
    _text(d, (80, y + 110), "This is a computer generated statement.", FONT, 22)
    img.save(out)
    return out


def make_pan_card(out: Path) -> Path:
    W, H = (1200, 760)
    img = Image.new("RGB", (W, H), (245, 240, 225))
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, W - 10, H - 10), outline=(60, 60, 60), width=4)
    _text(d, (40, 30), "INCOME TAX DEPARTMENT        GOVT. OF INDIA", FONT_BOLD, 30)
    _text(d, (40, 80), "Permanent Account Number Card", FONT, 24)
    d.line((40, 120, W - 40, 120), fill=(0, 0, 0), width=2)

    # photo placeholder box (deliberately not OCR-readable text)
    d.rectangle((880, 150, 1140, 450), outline=(120, 120, 120), width=3)
    _text(d, (1010, 290), "PHOTO", FONT, 26, fill=(150, 150, 150), anchor="mm")

    _text(d, (40, 160), "Permanent Account Number", FONT, 22)
    _text(d, (40, 195), "ABCDE1234F", FONT_MONO, 40)
    _text(d, (40, 280), "Name", FONT, 22)
    _text(d, (40, 315), "ARJUN MEHTA", FONT_BOLD, 36)
    _text(d, (40, 400), "Father's Name", FONT, 22)
    _text(d, (40, 435), "RAKESH MEHTA", FONT_BOLD, 32)
    _text(d, (40, 520), "Date of Birth", FONT, 22)
    _text(d, (40, 555), "14/08/1999", FONT_MONO, 34)
    _text(d, (40, 650), "Signature", FONT, 20)
    d.line((40, 690, 350, 690), fill=(0, 0, 0), width=2)
    img.save(out)
    return out


def make_salary_slip(out: Path) -> Path:
    img = Image.new("RGB", A4, (255, 255, 255))
    d = ImageDraw.Draw(img)
    _text(d, (80, 60), "INFOTECH SOLUTIONS PRIVATE LIMITED", FONT_BOLD, 36)
    _text(d, (80, 115), "Salary Slip for June 2026", FONT, 28)
    d.line((80, 165, 1574, 165), fill=(0, 0, 0), width=3)

    _text(d, (80, 200), "Employee Name: ARJUN MEHTA", FONT_BOLD, 30)
    _text(d, (900, 200), "Employee ID: EMP20451", FONT, 26)
    _text(d, (80, 250), "Designation: Software Engineer", FONT, 26)
    _text(d, (900, 250), "Pay Period: Jun-2026", FONT, 26)
    _text(d, (80, 300), "PF Number: KA/BNG/204518/000", FONT_MONO, 26)
    _text(d, (900, 300), "Bank A/c: 5010023456789012", FONT_MONO, 26)

    y = 400
    _text(d, (80, y), "Earnings", FONT_BOLD, 28)
    _text(d, (900, y), "Amount (INR)", FONT_BOLD, 28)
    d.line((80, y + 45, 1574, y + 45), fill=(0, 0, 0), width=2)
    earnings = [
        ("Basic", "42,500"),
        ("House Rent Allowance", "21,250"),
        ("Special Allowance", "16,500"),
        ("GROSS SALARY", "85,000"),
        ("Provident Fund (Deduction)", "-5,100"),
        ("Professional Tax (Deduction)", "-200"),
        ("NET SALARY", "79,700"),
    ]
    y += 65
    for label, amt in earnings:
        bold = label.isupper()
        _text(d, (80, y), label, FONT_BOLD if bold else FONT, 26)
        _text(d, (900, y), amt, FONT_BOLD if bold else FONT_MONO, 26)
        y += 55

    y += 30
    _text(d, (80, y), "Remarks: PAN deduction detail AJMPM4471K on file;", FONT, 24)
    y += 40
    _text(d, (80, y), "HR payroll queries: 9812345670", FONT, 24)
    y += 70
    _text(d, (80, y), "This is a system generated payslip.", FONT, 22)
    img.save(out)
    return out


def make_voter_id_card(out: Path) -> Path:
    """Elector's Photo Identity Card (EPIC) — a second ID layout.

    Same data as the PAN card on purpose: the pipeline must reach the same
    privacy verdict from a different arrangement of labels and values, and the
    voter number (3 letters + 7 digits) exercises `id_verification`'s
    `voter_id_number` partial spec, which no other sample doc reaches.
    """
    W, H = (1200, 780)
    img = Image.new("RGB", (W, H), (240, 244, 235))
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, W - 10, H - 10), outline=(60, 60, 60), width=4)
    _text(d, (40, 30), "ELECTION COMMISSION OF INDIA", FONT_BOLD, 30)
    _text(d, (40, 80), "Elector's Photo Identity Card", FONT, 24)
    d.line((40, 120, W - 40, 120), fill=(0, 0, 0), width=2)

    d.rectangle((880, 150, 1140, 460), outline=(120, 120, 120), width=3)
    _text(d, (1010, 300), "PHOTO", FONT, 26, fill=(150, 150, 150), anchor="mm")

    _text(d, (40, 160), "Elector's Name", FONT, 22)
    _text(d, (40, 195), "ARJUN MEHTA", FONT_BOLD, 36)
    _text(d, (40, 270), "Elector's Photo Identity Card No.", FONT, 22)
    _text(d, (40, 305), "ABC1234567", FONT_MONO, 38)
    _text(d, (40, 380), "Father's Name", FONT, 22)
    _text(d, (40, 415), "RAKESH MEHTA", FONT_BOLD, 32)
    _text(d, (40, 490), "Sex", FONT, 22)
    _text(d, (200, 490), "MALE", FONT, 26)
    _text(d, (40, 540), "Date of Birth", FONT, 22)
    _text(d, (40, 575), "14/08/1999", FONT_MONO, 34)
    _text(d, (40, 650), "Address", FONT, 22)
    _text(d, (40, 685), "Flat 12, MG Road, Bengaluru 560034", FONT, 26)
    img.save(out)
    return out


def make_marksheet(out: Path) -> Path:
    """Consolidated marksheet — the first document for `education_proof`.

    Carries one value per education type (institution, qualification, CGPA) plus
    the identifiers that must stay hidden (roll number, category, father's name,
    DOB), so the preset's keep/redact split is actually exercised by a layout.
    """
    img = Image.new("RGB", A4, (255, 255, 255))
    d = ImageDraw.Draw(img)
    _text(d, (80, 60), "UNIVERSITY OF BENGALURU", FONT_BOLD, 38)
    _text(d, (80, 118), "Consolidated Statement of Marks", FONT, 30)
    _text(d, (1100, 70), "Year of Passing: 2026", FONT, 24)
    d.line((80, 175, 1574, 175), fill=(0, 0, 0), width=3)

    _text(d, (80, 215), "Student Name: ARJUN MEHTA", FONT_BOLD, 30)
    _text(d, (900, 215), "Roll Number: 2023CS1042", FONT_MONO, 28)
    _text(d, (80, 265), "Father's Name: RAKESH MEHTA", FONT, 26)
    _text(d, (900, 265), "Date of Birth: 14/08/1999", FONT_MONO, 26)
    _text(d, (80, 315), "Programme: Bachelor of Engineering", FONT, 26)
    _text(d, (900, 315), "Category: GENERAL", FONT, 26)

    y = 400
    _text(d, (80, y), "Subject", FONT_BOLD, 26)
    _text(d, (800, y), "Credits", FONT_BOLD, 26)
    _text(d, (1050, y), "Grade", FONT_BOLD, 26)
    _text(d, (1300, y), "Marks", FONT_BOLD, 26)
    d.line((80, y + 45, 1574, y + 45), fill=(0, 0, 0), width=2)
    rows = [
        ("Data Structures", "4", "A", "88"),
        ("Operating Systems", "4", "A", "85"),
        ("Database Systems", "3", "A+", "91"),
        ("Computer Networks", "4", "B+", "78"),
        ("Software Engineering", "3", "A", "84"),
    ]
    y += 65
    for subject, credits, grade, marks in rows:
        _text(d, (80, y), subject, FONT, 24)
        _text(d, (800, y), credits, FONT, 24)
        _text(d, (1050, y), grade, FONT, 24)
        _text(d, (1300, y), marks, FONT_MONO, 24)
        y += 55
    d.line((80, y + 10, 1574, y + 10), fill=(0, 0, 0), width=2)

    y += 45
    _text(d, (80, y), "CGPA: 8.72", FONT_BOLD, 30)
    _text(d, (700, y), "Total Marks: 826", FONT_BOLD, 30)
    _text(d, (1200, y), "Percentage: 82.6", FONT_BOLD, 28)
    _text(d, (80, y + 70), "Qualification Awarded: Bachelor of Engineering", FONT, 26)
    _text(d, (80, y + 140), "Registrar queries: 9812345670", FONT, 24)
    _text(d, (80, y + 190), "This is a computer generated marksheet.", FONT, 22)
    img.save(out)
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for fn, name in ((make_bank_statement, "bank_statement.png"),
                     (make_pan_card, "pan_card.png"),
                     (make_salary_slip, "salary_slip.png"),
                     (make_voter_id_card, "voter_id_card.png"),
                     (make_marksheet, "marksheet.png")):
        path = out_dir / name
        fn(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

