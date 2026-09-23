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


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for fn, name in ((make_bank_statement, "bank_statement.png"),
                     (make_pan_card, "pan_card.png"),
                     (make_salary_slip, "salary_slip.png")):
        path = out_dir / name
        fn(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

