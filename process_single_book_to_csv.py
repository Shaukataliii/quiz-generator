# This script process the book (path) and saves the data as CSV file.

from src.module import BookProcessor, InputsHandler


def main():
    pdf_path, contain_images, use_ocr = INPUTS_HANDLER.take_evaluate_inputs_to_process_single_book()
    process_pdf_to_df(pdf_path, contain_images, use_ocr)
    
    

def process_pdf_to_df(pdf_path: str, contain_images: bool, use_ocr: bool):
    """Extracts details from book pdf path and saves them as pandas DataFrame.

    Args:
        pdf_path (str): Valid book pdf path.
        contain_images (bool): Whether to extract text from images.
        use_ocr (bool): Whether to use OCR.
    """
    if use_ocr:
        BOOK_PROCESSOR.use_ocr_to_process_book_pdf_and_save_df(pdf_path)
    else:
        BOOK_PROCESSOR.process_book_pdf_and_save_df(pdf_path, contain_images)


BOOK_PROCESSOR = BookProcessor()
if __name__ == "__main__":
    INPUTS_HANDLER = InputsHandler("classes_data_details.csv")
    main()
