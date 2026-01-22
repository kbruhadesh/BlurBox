#!/usr/bin/env python3

import os
import argparse
import re
import sys
import fitz  # pymupdf
import phonenumbers
from tqdm import tqdm
import cv2
import numpy as np

# --- CONFIGURATION & CONSTANTS ---
COLORS = {
    "white": fitz.pdfcolor["white"],
    "black": fitz.pdfcolor["black"],
    "red": fitz.pdfcolor["red"],
    "green": fitz.pdfcolor["green"],
    "blue": fitz.pdfcolor["blue"]
}

# Regex Patterns
PATTERNS = {
    'email': r"\S+@\S+\.\S+",
    'ssn': r"\b\d{3}-\d{2}-\d{4}\b",
    'creditcard': r"\b(?:\d[ -]*?){13,16}\b",
    'iban': r'\b[A-Z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?!(?:[ ]?[0-9]){3})(?:[ ]?[0-9]{1,2})?\b',
    'bic': r"\b[A-Z]{6}[A-Z0-9]{2}[A-Z0-9]{3}?\b",
    'timestamp': r"\b([0-1]?[0-9]|2[0-3]):[0-5][0-9]\b",
    'date': r"((?:[0]?[1-9]|[12][0-9]|3[01])(?:.?)(?:[./-]|[' '])(?:0?[1-9]|1[0-2]|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|Jan(?:uar)?|Feb(?:uar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)(?:[./-]|[' '])(?:[0-9]{4}|[0-9]{2}))"
}

class RedactionStats:
    def __init__(self):
        self.total_pages = 0
        self.redacted_pages = set()
        self.counts = {key: 0 for key in ['email', 'ssn', 'creditcard', 'iban', 'bic', 'timestamp', 'date', 'link', 'phonenumber', 'custom', 'barcode', 'qrcode']}

    def to_dict(self):
        return {
            'total_pages': self.total_pages,
            'redacted_pages': list(self.redacted_pages),
            'redacted_items': self.counts
        }

class Redactor:
    def __init__(self, options):
        self.options = options
        self.color = COLORS.get(options.get('color', 'black'), COLORS['black'])
        self.text = options.get('text', '[REDACTED]')
        self.stats = RedactionStats()

    def _get_regex_matches(self, text, pattern):
        return list(set(re.findall(pattern, text, flags=re.IGNORECASE)))

    def _get_phonenumbers(self, text):
        return [match.raw_string for match in phonenumbers.PhoneNumberMatcher(text, None)]

    def _find_visual_codes(self, page, code_type):
        """
        Find barcodes or QR codes on the page using OpenCV.
        Returns a list of fitz.Rect objects.
        """
        rects = []
        zoom = 3
        # Get pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
        
        # Convert to numpy array (RGB) then to BGR for OpenCV
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        cv2_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        qr_detector = cv2.QRCodeDetector()
        barcode_detector = None
        try:
             barcode_detector = cv2.barcode.BarcodeDetector()
        except AttributeError:
            pass

        points_list = []
        
        # QR Codes
        if code_type == 'qrcode':
            retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(cv2_img)
            if retval and points is not None:
                points_list.extend(points)

        # Barcodes
        elif code_type == 'barcode' and barcode_detector:
             retval, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(cv2_img)
             if retval and points is not None:
                  points_list.extend(points)

        for point_set in points_list:
             point_set = point_set.astype(np.int32)
             rect = cv2.boundingRect(point_set)
             x_px, y_px, w_px, h_px = rect
             
             x0 = x_px / zoom
             y0 = y_px / zoom
             x1 = (x_px + w_px) / zoom
             y1 = (y_px + h_px) / zoom
             rects.append(fitz.Rect(x0, y0, x1, y1))

        return rects

    def process_page(self, page, page_num):
        text = page.get_text("text")
        redaction_rects = []
        
        # 1. Regex Redactions
        for key, pattern in PATTERNS.items():
            if self.options.get(key):
                matches = self._get_regex_matches(text, pattern)
                if matches:
                    print(f" |  Found {len(matches)} {key} matches on Page {page_num+1}")
                    self.stats.counts[key] += len(matches)
                    for match in matches:
                        redaction_rects.extend(page.search_for(match))

        # 2. Phone Numbers
        if self.options.get('phonenumber'):
            matches = self._get_phonenumbers(text)
            if matches:
                print(f" |  Found {len(matches)} phone number matches on Page {page_num+1}")
                self.stats.counts['phonenumber'] += len(matches)
                for match in matches:
                    redaction_rects.extend(page.search_for(match))

        # 3. Links
        if self.options.get('link'):
            links = page.get_links()
            if links:
                print(f" |  Found {len(links)} links on Page {page_num+1}")
                self.stats.counts['link'] += len(links)
                for link in links:
                    redaction_rects.append(link['from'])

        # 4. Custom Mask
        if self.options.get('custom') and self.options.get('mask'):
            mask = self.options.get('mask')
            matches = self._get_regex_matches(text, r'\b' + re.escape(mask) + r'\b')
            if matches:
                print(f" |  Found {len(matches)} custom matches on Page {page_num+1}")
                self.stats.counts['custom'] += len(matches)
                for match in matches:
                     redaction_rects.extend(page.search_for(match))

        # 5. Visual Codes (Barcode/QR)
        if self.options.get('barcode'):
            rects = self._find_visual_codes(page, 'barcode')
            if rects:
                print(f" |  Found {len(rects)} barcodes on Page {page_num+1}")
                self.stats.counts['barcode'] += len(rects)
                redaction_rects.extend(rects)

        if self.options.get('qrcode'):
            rects = self._find_visual_codes(page, 'qrcode')
            if rects:
                 print(f" |  Found {len(rects)} QR codes on Page {page_num+1}")
                 self.stats.counts['qrcode'] += len(rects)
                 redaction_rects.extend(rects)

        # Apply Redactions
        if redaction_rects:
            self.stats.redacted_pages.add(page_num)
            for rect in redaction_rects:
                page.add_redact_annot(
                    quad=rect,
                    text=self.text,
                    text_color=COLORS['white'],
                    fill=self.color,
                    cross_out=True
                )
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    def process_file(self, input_path, output_path):
        print(f"\n[i] Processing file: {input_path}")
        try:
            doc = fitz.open(input_path)
            self.stats.total_pages = len(doc)
            
            # Determine which pages to process
            pages_arg = self.options.get('pages')
            if pages_arg is not None:
                pages_to_process = [p for p in pages_arg if 0 <= p < len(doc)]
            else:
                pages_to_process = range(len(doc))

            for page_num in tqdm(pages_to_process, desc="Redacting Pages", unit="page"):
                page = doc.load_page(page_num)
                self.process_page(page, page_num)

            print(f"[i] Saving to: {output_path}")
            doc.ez_save(output_path)
            return self.stats.to_dict()

        except Exception as e:
            print(f"[Error] Failed to process {input_path}: {e}")
            return None

def is_directory(path):
    return os.path.isdir(path)

def blurbox_redact(input_path, output_path, options):
    """
    API entry point for Flask app.
    """
    redactor = Redactor(options)
    
    # Handle directory vs file
    if is_directory(input_path):
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
            
        results = []
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                file_in = os.path.join(input_path, filename)
                file_out = os.path.join(output_path, f"redacted_{filename}") if output_path else file_in.replace('.pdf', '_redacted.pdf')
                stats = redactor.process_file(file_in, file_out)
                results.append(stats)
        return results[-1] if results else None # Return last stats for compatibility or aggregate? 
        # API usually expects single file stats for the web app, 
        # but if we ever support batch upload in web app, we'd need to adjust.
        # For now, app.py sends single file path.
    else:
        return redactor.process_file(input_path, output_path)

def main():
    print("""BlurBox @Commit STROM""")
    
    parser = argparse.ArgumentParser(description="PDF Redactor")
    parser.add_argument('-i', '--input', required=True, help='Input PDF file or directory')
    parser.add_argument('-o', '--output', help='Output PDF file or directory')
    
    # Flags
    parser.add_argument('-e', '--email', action='store_true', help='Redact emails')
    parser.add_argument('-p', '--phonenumber', action='store_true', help='Redact phone numbers')
    parser.add_argument('-l', '--link', action='store_true', help='Redact links')
    parser.add_argument('-d', '--date', action='store_true', help='Redact dates')
    parser.add_argument('-f', '--timestamp', action='store_true', help='Redact timestamps')
    parser.add_argument('-s', '--iban', action='store_true', help='Redact IBANs')
    parser.add_argument('-b', '--bic', action='store_true', help='Redact BICs')
    parser.add_argument('-r', '--barcode', action='store_true', help='Redact Barcodes')
    parser.add_argument('-q', '--qrcode', action='store_true', help='Redact QR Codes')
    parser.add_argument('--ssn', action='store_true', help='Redact SSNs')
    parser.add_argument('--creditcard', action='store_true', help='Redact Credit Cards')
    
    parser.add_argument('-m', '--mask', type=str, help='Custom word to redact')
    parser.add_argument('-t', '--text', type=str, default='[REDACTED]', help='Redaction text')
    parser.add_argument('-c', '--color', type=str, default='black', choices=COLORS.keys(), help='Redaction color')
    
    args = parser.parse_args()
    
    options = vars(args)
    options['custom'] = bool(args.mask) # Set custom flag if mask is provided
    
    # Check output path validity
    if is_directory(args.input) and args.output and not is_directory(args.output):
         # If output doesn't exist, create it as dir
         if not args.output.lower().endswith('.pdf'):
             os.makedirs(args.output, exist_ok=True)
         else:
             print("[Error] Input is directory, output must be directory.")
             sys.exit(1)
             
    if not is_directory(args.input) and args.output and is_directory(args.output):
         # Single file input, dir output -> construct filename
         filename = os.path.basename(args.input)
         args.output = os.path.join(args.output, f"redacted_{filename}")

    blurbox_redact(args.input, args.output if args.output else args.input.replace('.pdf', '_redacted.pdf'), options)

if __name__ == "__main__":
    main()
