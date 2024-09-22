from langchain_codebase.codebase import *
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract
import re


QUIZ_GENERATOR_PROMPT = """
You will receive the contents of a particular chapter of a student's book. Your job is to create {no_questions} multiple choice question answers using those contents and provide them in the form of python code where:
1- All are questions are inside a list.
2- Each question is a dictionary of values.
3- The keys of the question dictionary should be:
question (representing question as str)
choices (list of choices representing each choice as str)
correct_choice (the index of correct choice i.e. the index of correct choice form the choices list.)

Note: Strictly follow the rules and don't provide anything else than the asked. Only provide a clean list ready to process further.

contents: {book_content}
"""
PARAMS_FILEPATH = r'src\params.yaml'

class DocumentProcessor:
    UNIT_REGEX = r'Unit\s+(\d+)'

    def extract_details_from_docs_as_df(self, docs, subject, book_class):
        book_data = {
            'class': [],
            'subject': [],
            'page_no': [],
            'unit_no': [],
            'content': []
        }
        for doc in docs:
            page_no, unit_no, content = self.extract_page_no_unit_no_page_content_from_doc(doc)
            book_data['class'].append(book_class)
            book_data['subject'].append(subject)
            book_data['page_no'].append(page_no)
            book_data['unit_no'].append(unit_no)
            book_data['content'].append(content)

        book_data_df = pd.DataFrame(book_data)
        self.check_for_missing_unit_and_content_vals(book_data_df)
        
        return book_data_df

    def extract_page_no_unit_no_page_content_from_doc(self, doc):
        page_no = doc.metadata['page']
        page_content = doc.page_content
        # print(f"\nPage no: {page_no}, content length is: {len(page_content)}. Extracting unit no...")
        unit_no = self.extract_unit_no_from_content(page_content)
        return (page_no, unit_no, page_content)

    def extract_unit_no_from_content(self, page_content: str):
        """Using regex and returns Unit no. numerical_value is found else "". """
        match = re.search(self.UNIT_REGEX, page_content)
        if match:
            return match.group(1)
        else:
            return ""
    

    def check_for_missing_unit_and_content_vals(self, df: pd.DataFrame):
        """Checks how many rows don't have unit no. or content information. If all rows don't have, then raises exception."""
        num_rows = df.shape[0]
        pages_with_no_unit_info = 0
        pages_with_no_content = 0

        for _, row in df.iterrows():
            unit_no = row['unit_no'].strip()
            content = row['content']
            if not unit_no.isnumeric():
                print(f"Unit no: {unit_no} is not valid.")
                pages_with_no_unit_info += 1
            if not content:
                pages_with_no_content += 1
        
        print(f"\nTotal pages: {num_rows}")
        print(f"Pages with no unit info: {pages_with_no_unit_info}")
        print(f"Pages with no content: {pages_with_no_content}")

        if (pages_with_no_unit_info == num_rows):
            raise Exception("Unit no. not available on any page.")
        if (pages_with_no_content == num_rows):
            raise Exception("Content not present on any page.")



class BookProcessor:
    csv_dirname = "books_df_csvs"

    def process_book_pdf_and_save_df(self, book_pdf_path: str, contain_images: bool = False):
        """Extracts book_pdf_name, subject and book_class from the provided path. Uses them to process extracts details as DataFrame and saves them using book_pdf_name."""
        self.set_book_path_detail_vars(book_pdf_path)
        df = self.extract_details_from_book_pdf_path_as_df(contain_images)
        self.save_book_df(df, self.book_file_name)
    
    def extract_details_from_book_pdf_path_as_df(self, contain_images: bool = False):
        if is_valid_pdf_path(self.book_pdf_path):
            docs = load_single_pdf(self.book_pdf_path, contain_images)
            df = DOC_PROCESSOR.extract_details_from_docs_as_df(docs, self.subject, self.book_class)
            return df


    def use_ocr_to_process_book_pdf_and_save_df(self, book_pdf_path: str):
        """Uses OCR to process the pdf.
        Extracts book_pdf_name, subject and book_class from the provided path. Uses them to process extracts details as DataFrame and saves them using book_pdf_name."""
        self.set_book_path_detail_vars(book_pdf_path)
        df = self.use_ocr_to_extract_details_from_book_pdf_path_as_df()
        self.save_book_df(df)

    def use_ocr_to_extract_details_from_book_pdf_path_as_df(self):
        if is_valid_pdf_path(self.book_pdf_path):
            docs = OCR_USER.convert_pdf_path_to_docs(self.book_pdf_path, self.book_pdf_name)
            df = DOC_PROCESSOR.extract_details_from_docs_as_df(docs, self.subject, self.book_class)
            return df
        
    
    def set_book_path_detail_vars(self, book_pdf_path: str):
        self.book_pdf_path = book_pdf_path
        self.book_pdf_name, self.subject, self.book_class = BOOK_PATH_PROCESSOR.extract_filename_subject_class_from_book_pdffile_path(book_pdf_path)
        self.book_name = (self.subject + " " + self.book_class)


    def save_book_df(self, df: pd.DataFrame):
        if not self.csv_files_dir_exist():
            os.makedirs(self.csv_dirname, exist_ok=True)

        book_save_path = os.path.join(self.csv_dirname, self.book_name + ".csv")
        df.to_csv(book_save_path, index=False)
        print(f"Book saved as: {book_save_path}")
        
    def csv_files_dir_exist(self):
        if os.path.exists(self.csv_dirname):
            return True
        else:
            return False


class BookPathProcessor:
    def extract_filename_subject_class_from_book_pdffile_path(self, book_pdf_path: str):
        self.book_pdf_path = book_pdf_path

        self.book_pdf_name = self.extract_book_pdffile_name_from_path()
        subject, book_class = self.extract_subject_and_class_from_book_pdffile_name()
        return self.book_pdf_name, subject, book_class

    def extract_book_pdffile_name_from_path(self):
        book_pdf_name = os.path.basename(self.book_pdf_path)
        return book_pdf_name

    def extract_subject_and_class_from_book_pdffile_name(self):
        """Book pdffile name is supposed to be like: 'Physics 9.pdf', so it separates using space and return the values. The class name should be isnumeric."""
        book_name = self.book_pdf_name.rsplit(".", 1)[0]
        if not ' ' in book_name:
            raise Exception(f"Book name doesn't contain any space. It is: {book_name}")
        
        if not self.name_ends_with_numeric_char(book_name):
            raise Exception(f"Book name doesn't class i.e. it doesn't end with a numeric character.")
        
        subject, book_class = book_name.rsplit(" ", 1)
        return (subject, book_class)
    
    def name_ends_with_numeric_char(self, name: str):
        end_char = name[-1]
        return True if end_char.isnumeric() else False



class OCRUser:
    def __init__(self):
        params = load_yaml_file(PARAMS_FILEPATH)
        self.poppler_path = params['poppler_path']

    def convert_pdf_path_to_docs(self, book_pdf_path: str, book_pdf_name: str):
        self.book_pdf_path = book_pdf_path
        self.book_pdf_name = book_pdf_name

        pages = self.convert_pdf_path_to_images()
        docs = self.convert_pdf_images_to_docs(pages)
        return docs

    def convert_pdf_path_to_images(self):
        images = convert_from_path(pdf_path = self.book_pdf_path, poppler_path = self.poppler_path)
        return images


    def convert_pdf_images_to_docs(self, pdf_images):
        docs = []
        for image_no, image in enumerate(pdf_images):
            image_no = (image_no + 1)
            image_doc = self.convert_pdf_image_to_doc(image, image_no)
            docs.append(image_doc)
        return docs

    def convert_pdf_image_to_doc(self, pdf_image: str, image_no: int):
        text = pytesseract.image_to_string(pdf_image)
        doc = Document(page_content=text)
        doc.metadata['page'] = image_no
        doc.metadata['source'] = self.book_pdf_name
        return doc




class InputsHandler:
    def take_evaluate_inputs_to_process_single_book(self):
        """Takes pdf_path, contain_images, use_ocr input parameters, evalutes them and on successful evaluation.
        returns (pdf_path: str, contain_images: bool, use_ocr: bool)."""
        self.pdf_path = input("Enter book pdf file path: ")
        if is_valid_pdf_path(self.pdf_path):
            return self.take_other_inputs_and_process_them()

    def take_other_inputs_and_process_them(self):
        contain_images = input("Extract details from images (yes/no): ")
        use_ocr = input("Use OCR? (yes/no) ")
        sure = input("Are you sure? (yes/no) ")

        sure = self.evaluate_convert_val_to_bool(sure)
        use_ocr = self.evaluate_convert_val_to_bool(use_ocr)
        contain_images = self.evaluate_convert_val_to_bool(contain_images)

        if sure:
            return (self.pdf_path, contain_images, use_ocr)
        else:
            return self.handle_pdfpath_input()

    def evaluate_convert_val_to_bool(self, value: str):
        """The value can only be yes or no. Transforms to True if yes else False."""
        if value not in ['yes', 'no']:
            raise Exception(f"Invalid Value. Expected yes or no. Got: {value}")
        
        value = (value == 'yes')
        return value




class Utils:
    def load_ollama_model(model: str = 'gemma2:2b'):
        return OllamaLLM(model=model)

    def get_response_from_model(llm, prompt):
        return llm.invoke(prompt)

    def get_quiz_generator_prompt():
        prompt = PromptTemplate(
            input_variables = ['no_questions', 'book_content'],
            template = QUIZ_GENERATOR_PROMPT
        )
        return prompt


DOC_PROCESSOR = DocumentProcessor()
OCR_USER = OCRUser()
BOOK_PATH_PROCESSOR = BookPathProcessor()