"""
Document conversion utility using LibreOffice.
Converts documents to PDF with configurable timeout.
"""
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple


# Timeout for LibreOffice conversion (in seconds)
# Changed from 5s to 60s (1 minute) to allow for larger documents
CONVERSION_TIMEOUT_SECONDS = 60


class DocumentConversionError(Exception):
    """Raised when document conversion fails."""
    pass


def convert_to_pdf(
    input_path: str,
    output_dir: Optional[str] = None,
    timeout: int = CONVERSION_TIMEOUT_SECONDS
) -> str:
    """
    Convert a document to PDF using LibreOffice.
    
    Args:
        input_path: Path to the input document
        output_dir: Directory for the output PDF (defaults to same directory as input)
        timeout: Timeout in seconds (default: 60 seconds)
    
    Returns:
        Path to the generated PDF file
    
    Raises:
        DocumentConversionError: If conversion fails
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise DocumentConversionError(f"Input file does not exist: {input_path}")
    
    if output_dir is None:
        output_dir = str(input_path.parent)
    
    # List of LibreOffice command variants to try
    libreoffice_commands = ["soffice", "libreoffice"]
    
    errors: List[Tuple[str, str]] = []
    
    for cmd in libreoffice_commands:
        try:
            result = subprocess.run(
                [
                    cmd,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", output_dir,
                    str(input_path)
                ],
                capture_output=True,
                text=True,
                timeout=timeout  # 60 seconds (1 minute) timeout
            )
            
            if result.returncode == 0:
                # Construct expected output path
                output_filename = input_path.stem + ".pdf"
                output_path = Path(output_dir) / output_filename
                
                if output_path.exists():
                    return str(output_path)
                else:
                    errors.append((cmd, f"Output file not created: {output_path}"))
            else:
                errors.append((cmd, f"Exit code {result.returncode}: {result.stderr}"))
                
        except subprocess.TimeoutExpired:
            errors.append((cmd, f"timed out after {timeout}s"))
        except FileNotFoundError:
            errors.append((cmd, "command not found"))
        except Exception as e:
            errors.append((cmd, str(e)))
    
    # All commands failed
    error_messages = "; ".join([f"{cmd}: {err}" for cmd, err in errors])
    raise DocumentConversionError(f"Failed to convert to PDF. Errors: {error_messages}")


def convert_document_to_pdf(
    document_id: str,
    input_path: str,
    output_dir: Optional[str] = None,
    timeout: int = CONVERSION_TIMEOUT_SECONDS
) -> str:
    """
    Convert a document to PDF with document ID tracking.
    
    Args:
        document_id: Unique identifier for the document
        input_path: Path to the input document
        output_dir: Directory for the output PDF
        timeout: Timeout in seconds (default: 60 seconds / 1 minute)
    
    Returns:
        Path to the generated PDF file
    
    Raises:
        DocumentConversionError: If conversion fails
    """
    try:
        return convert_to_pdf(input_path, output_dir, timeout)
    except DocumentConversionError as e:
        raise DocumentConversionError(
            f"Failed to convert document {document_id} to PDF: {e}"
        ) from e


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_conversion.py <input_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = convert_to_pdf(input_file, output_directory)
        print(f"Successfully converted to: {output_path}")
    except DocumentConversionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
