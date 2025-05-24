import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from pdf_redactor import blurbox_redact

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Required for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'

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

def redact_pdf(input_path, output_path, redact_options, color, text, mask):
    """Redact PDF with specified options using blurbox_redact"""
    try:
        # Prepare options dict for blurbox_redact, including all possible options with defaults
        options = {
            'input': input_path,
            'output': output_path,
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
        blurbox_redact(input_path, output_path, options)
        redaction_info = {
            'total_pages': 'N/A',
            'redacted_items': {},
            'redacted_pages': []
        }
        return True, redaction_info
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
            
        if not file.filename.lower().endswith('.pdf'):
            flash('Please upload a PDF file')
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
        
        # Process PDF
        success, result = redact_pdf(input_path, output_path, redact_options, color, text, mask)
        
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