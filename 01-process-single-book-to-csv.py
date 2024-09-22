from src.module import BookProcessor, InputsHandler


def main():
    pdf_path, contain_images, use_ocr = INPUTS_HANDLER.take_evaluate_inputs_to_process_single_book()
    
    if use_ocr:
        BOOK_PROCESSOR.use_ocr_to_process_book_pdf_and_save_df(pdf_path)
    else:
        BOOK_PROCESSOR.process_book_pdf_and_save_df(pdf_path, contain_images)


if __name__ == "__main__":
    BOOK_PROCESSOR = BookProcessor()
    INPUTS_HANDLER = InputsHandler()
    main()
