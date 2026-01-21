# BLURBOX - PDF Redaction Tool

BLURBOX is a powerful web-based PDF redaction tool that helps you protect sensitive information in your PDF documents. The application provides an intuitive interface for redacting various types of sensitive data, including phone numbers, email addresses, links, dates, and more.

## Features

- **Web Interface**: Modern, easy-to-use web interface for uploading and redacting PDFs
- **Live Preview**: View the redacted PDF directly in the browser before downloading.
- **Page Selection**: Specify exact pages or ranges to redact (e.g., "1-3, 5").
- **Multiple Redaction Types**:
  - Phone Numbers
  - Email Addresses
  - URLs/Links
  - Dates
  - Social Security Numbers
  - Credit Card Numbers
  - Custom Words/Phrases
- **Customization Options**:
  - Choose redaction color (black, red, blue, green)
  - Custom redaction text
  - Custom word/phrase redaction
- **Real-time Processing**: Immediate feedback on redaction process
- **Detailed Summary**: View accurate redaction statistics (pages, items) after processing
- **Secure**: Files are processed locally and automatically cleaned up

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kbruhadesh/blurbx.git
   cd blurbx
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: No external system dependencies are required anymore. Barcode/QR code detection uses OpenCV.*

3. Create an uploads directory (if not exists):
   ```bash
   mkdir uploads
   ```

## Usage

1. Start the web application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Upload your PDF file.

4. (Optional) Specify a **Page Range** (e.g., `1-5`) in the Scope section.

5. Select the types of information you want to redact.

6. Click "Process & Redact".

7. Preview the result in the browser and download the redacted PDF.

## Requirements

- Python 3.8 or higher
- Flask
- PyMuPDF (fitz)
- phonenumbers
- Pillow
- OpenCV (opencv-python)
- numpy
- tqdm

## Project Structure

```
blurbox/
├── app.py              # Flask web application
├── pdf_redactor.py     # PDF redaction core functionality
├── requirements.txt    # Python dependencies
├── templates/         # HTML templates
│   └── index.html     # Main web interface
└── uploads/          # Temporary storage for uploaded files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with Flask
- PDF processing powered by PyMuPDF
- UI components from Bootstrap
