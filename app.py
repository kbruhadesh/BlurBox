import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from pdf_redactor import blurbox_redact

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Redaction options
REDACT_OPTIONS = [
    ('phonenumber', 'Phone Numbers'),
    ('email', 'Email Addresses'),
    ('link', 'URLs/Links'),
    ('date', 'Dates'),
    ('ssn', 'Social Security Numbers'),
    ('creditcard', 'Credit Card Numbers'),
    ('custom', 'Custom Word/Phrase')
]

# Color options
COLOR_OPTIONS = ['black', 'red', 'blue', 'green']

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/preview/<filename>')
def preview_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

def parse_page_range(page_range_str):
    """Parse string like '1-3, 5' into list of 0-indexed integers."""
    if not page_range_str or not page_range_str.strip():
        return None
    
    pages = set()
    parts = page_range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                pages.update(range(start - 1, end)) # 0-indexed
            except ValueError:
                continue
        else:
            try:
                pages.add(int(part) - 1) # 0-indexed
            except ValueError:
                continue
    return list(pages)

def redact_pdf(input_path, output_path, redact_options, color, text, mask, page_range_str=None):
    """Redact PDF with specified options using blurbox_redact"""
    try:
        pages = parse_page_range(page_range_str)
        # Prepare options dict for blurbox_redact
        options = {
            'input': input_path,
            'output': output_path,
            'pages': pages,
            'phonenumber': redact_options.get('phonenumber', False),
            'email': redact_options.get('email', False),
            'link': redact_options.get('link', False),
            'date': redact_options.get('date', False),
            'ssn': redact_options.get('ssn', False),
            'creditcard': redact_options.get('creditcard', False),
            'custom': redact_options.get('custom', False),
            'mask': mask if mask else None,
            'color': color,
            'text': text,
            'iban': False,
            'bic': False,
            'timestamp': False,
            'barcode': False,
            'qrcode': False
        }
        stats = blurbox_redact(input_path, output_path, options)
        return True, stats
    except Exception as e:
        return False, str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
            
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        if not file.filename.lower().endswith('.pdf') or file.mimetype != 'application/pdf':
            flash('Invalid file. Please upload a valid PDF.')
            return redirect(request.url)
            
        # Save uploaded file
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(input_path)
        
        # Generate output filename
        output_filename = f"redacted_{secure_filename(file.filename)}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Get redaction options
        redact_options = {
            'phonenumber': 'phonenumber' in request.form,
            'email': 'email' in request.form,
            'link': 'link' in request.form,
            'date': 'date' in request.form,
            'ssn': 'ssn' in request.form,
            'creditcard': 'creditcard' in request.form,
            'custom': 'custom' in request.form
        }
        
        # Get other options
        color = request.form.get('color', 'black')
        text = request.form.get('text', '[REDACTED]')
        mask = request.form.get('mask', '')
        page_range_str = request.form.get('pages', '')
        
        # Process PDF
        success, result = redact_pdf(input_path, output_path, redact_options, color, text, mask, page_range_str)
        
        if success:
            # Clean up input file
            os.remove(input_path)
            return render_template('index.html', 
                                redact_options=REDACT_OPTIONS,
                                color_options=COLOR_OPTIONS,
                                redaction_info=result,
                                download_file=output_filename)
        else:
            flash(f'Error processing PDF: {result}')
            return redirect(request.url)
            
    return render_template('index.html', 
                         redact_options=REDACT_OPTIONS,
                         color_options=COLOR_OPTIONS)

if __name__ == "__main__":
    app.run(debug=True) 