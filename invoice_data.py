import streamlit as st
import google.generativeai as genai
import json
import os
from pathlib import Path
from datetime import datetime
import base64
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(page_title="Invoice Data Extractor", layout="wide")

# Initialize session state
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Configure Gemini API
@st.cache_resource
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in secrets or environment variables")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

# Create results directory
def ensure_results_dir():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

# High-quality extraction prompt
EXTRACTION_PROMPT = """
You are an expert invoice data extraction system. Analyze the provided invoice image and extract ALL requested information with high precision and accuracy.

IMPORTANT INSTRUCTIONS:
1. Extract data exactly as it appears in the invoice
2. For numeric values, remove currency symbols and extra characters, keep only numbers and decimals
3. For phone numbers, extract all numbers including country codes and formatting
4. If a field is not present or visible, use null
5. For items, if a value is not specified in the invoice, use 0 for numeric fields
6. Ensure all monetary amounts are accurate to 2 decimal places
7. Extract item_total_amount as: (quantity * unit_price) - discount + tax

EXTRACT THE FOLLOWING INFORMATION AND RETURN ONLY VALID JSON:

{
  "shop_info": {
    "shop_name": "Extract the business/shop name",
    "shop_address": "Complete address including street, city, state, postal code",
    "shop_contact_numbers": ["List", "all", "phone", "numbers"],
    "shop_email": "Email address if present, else null"
  },
  "invoice_details": {
    "receipt_number": "Receipt or transaction number if available",
    "invoice_number": "Invoice number or bill number",
    "invoice_date": "Date in YYYY-MM-DD format",
    "invoice_subtotal": "Subtotal before tax and discount (numeric only)",
    "invoice_total": "Final total amount (numeric only)",
    "invoice_total_discount": "Total discount amount (numeric only, 0 if not present)",
    "item_count": "Total number of unique items"
  },
  "line_items": [
    {
      "item_code": "Product code/SKU if available, else null",
      "item_name": "Product/item name",
      "quantity": "Quantity as number",
      "unit_price": "Price per unit (numeric only)",
      "discount": "Item discount amount (numeric only, 0 if not present)",
      "tax": "Item tax amount (numeric only, 0 if not present)",
      "item_total_amount": "Total for this line item including tax and discount"
    }
  ]
}

CRITICAL REQUIREMENTS:
- Return ONLY valid JSON, no additional text or explanation
- All numeric values must be numbers (not strings)
- If any field is missing, use null (not empty string)
- Ensure JSON is properly formatted and valid
- Double-check all arithmetic calculations
- Extract all visible items from the invoice"""

# Extract data from invoice using Gemini
def extract_invoice_data(image_data):
    try:
        model = setup_gemini()
        
        # Convert image to base64 for API
        if isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            buffer = io.BytesIO()
            image_data.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        
        # Prepare image for Gemini
        image_part = {
            "mime_type": "image/png",
            "data": base64.standard_b64encode(image_bytes).decode()
        }
        
        # Send to Gemini
        response = model.generate_content([EXTRACTION_PROMPT, image_part])
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        extracted_data = json.loads(response_text.strip())
        return extracted_data
    
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {e}")
        return None
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return None

# Save extracted data to JSON file
def save_extraction_result(data, filename=None):
    results_dir = ensure_results_dir()
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        invoice_num = data.get("invoice_details", {}).get("invoice_number", "unknown").replace("/", "_")
        filename = f"invoice_{invoice_num}_{timestamp}.json"
    
    filepath = results_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath

# Display extracted data
def display_extracted_data(data):
    if not data:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Shop Information")
        shop_info = data.get("shop_info", {})
        st.write(f"**Name:** {shop_info.get('shop_name', 'N/A')}")
        st.write(f"**Address:** {shop_info.get('shop_address', 'N/A')}")
        
        contacts = shop_info.get('shop_contact_numbers', [])
        if contacts:
            st.write(f"**Phone:** {', '.join(contacts)}")
        
        email = shop_info.get('shop_email')
        if email:
            st.write(f"**Email:** {email}")
    
    with col2:
        st.subheader("📄 Invoice Details")
        invoice_details = data.get("invoice_details", {})
        st.write(f"**Invoice #:** {invoice_details.get('invoice_number', 'N/A')}")
        st.write(f"**Receipt #:** {invoice_details.get('receipt_number', 'N/A')}")
        st.write(f"**Date:** {invoice_details.get('invoice_date', 'N/A')}")
        st.write(f"**Item Count:** {invoice_details.get('item_count', 'N/A')}")
    
    st.subheader("💰 Financial Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Subtotal", f"${invoice_details.get('invoice_subtotal', 0):.2f}")
    with col2:
        st.metric("Total Discount", f"${invoice_details.get('invoice_total_discount', 0):.2f}")
    with col3:
        st.metric("Total Tax", f"${sum(item.get('tax', 0) for item in data.get('line_items', [])):.2f}")
    with col4:
        st.metric("TOTAL", f"${invoice_details.get('invoice_total', 0):.2f}", delta=None)
    
    # Display line items
    if data.get("line_items"):
        st.subheader("📦 Line Items")
        
        items_data = []
        for idx, item in enumerate(data.get("line_items", []), 1):
            items_data.append({
                "#": idx,
                "Code": item.get("item_code", "N/A"),
                "Item Name": item.get("item_name", "N/A"),
                "Qty": item.get("quantity", 0),
                "Unit Price": f"${item.get('unit_price', 0):.2f}",
                "Discount": f"${item.get('discount', 0):.2f}",
                "Tax": f"${item.get('tax', 0):.2f}",
                "Total": f"${item.get('item_total_amount', 0):.2f}"
            })
        
        st.dataframe(items_data, use_container_width=True)

# Main app
st.title("🧾 Invoice Data Extraction System")
st.markdown("Extract structured data from invoice images using AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Upload an invoice image to extract data")

# Main content
tab1, tab2 = st.tabs(["Upload & Extract", "View Results"])

with tab1:
    st.subheader("Upload Invoice Image")
    
    uploaded_file = st.file_uploader(
        "Choose an invoice image",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        help="Supported formats: JPG, PNG, BMP, GIF"
    )
    
    if uploaded_file:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(st.session_state.uploaded_image, caption="Uploaded Invoice", use_container_width=True)
        
        with col2:
            if st.button("🔍 Extract Data", use_container_width=True, type="primary"):
                with st.spinner("Extracting invoice data..."):
                    extraction_result = extract_invoice_data(st.session_state.uploaded_image)
                    
                    if extraction_result:
                        st.session_state.extraction_result = extraction_result
                        
                        # Save to file
                        filepath = save_extraction_result(extraction_result)
                        st.success(f"✅ Data extracted successfully!")
                        st.info(f"📁 Saved to: `{filepath}`")
                        
                        # Display results
                        display_extracted_data(extraction_result)
                        
                        # Download options
                        st.subheader("📥 Download Results")
                        json_str = json.dumps(extraction_result, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="Download as JSON",
                            data=json_str,
                            file_name=f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

with tab2:
    st.subheader("📋 Extraction Results")
    
    results_dir = ensure_results_dir()
    results_files = list(results_dir.glob("*.json"))
    
    if results_files:
        # Sort by modification time, newest first
        results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        selected_file = st.selectbox(
            "Select a result file",
            results_files,
            format_func=lambda x: x.name
        )
        
        if selected_file:
            with open(selected_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            st.write(f"**File:** {selected_file.name}")
            st.write(f"**Modified:** {datetime.fromtimestamp(selected_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.divider()
            
            display_extracted_data(data)
            
            # Raw JSON viewer
            with st.expander("📄 Raw JSON Data"):
                st.json(data)
    else:
        st.info("📭 No extraction results found. Upload an invoice and extract data to see results here.")