from langchain_codebase.codebase import *
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


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

class DocumentProcessor:
    UNIT_WORD = "Unit "
    EXTRA_CHARS_TO_EXTRACT_AROUND_UNIT_WORD = 20
    UNIT_VAL_ENDING_VALS = [":", "\n"]

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

    def extract_unit_no_from_content(self, content: str):
        chunk = self.extract_chunk_containing_unit_value(content)

        unit_val_start_index = self.extract_unit_val_start_index(chunk)
        unit_val_end_index = self.extract_unit_val_end_index(chunk)    

        unit_val = chunk[unit_val_start_index:unit_val_end_index]
        return unit_val

    def extract_chunk_containing_unit_value(self, content: str):
        unit_word_index_in_content = content.find(self.UNIT_WORD)

        if self.is_found_invalid_index(unit_word_index_in_content):
            return ""
        
        chunk_start_index = self.get_chunk_start_index(unit_word_index_in_content)
        chunk_end_index = self.get_chunk_end_index(unit_word_index_in_content)

        chunk = content[chunk_start_index:chunk_end_index]
        return chunk

    def get_chunk_start_index(self, unit_word_index):
        if self.required_extra_chars_available_before_unit_word_index(unit_word_index):
            chunk_start_index = unit_word_index - self.EXTRA_CHARS_TO_EXTRACT_AROUND_UNIT_WORD
        else:
            chunk_start_index = unit_word_index
        return chunk_start_index

    def required_extra_chars_available_before_unit_word_index(self, unit_word_index):
        if unit_word_index > self.EXTRA_CHARS_TO_EXTRACT_AROUND_UNIT_WORD:
            return True
        else:
            return False

    def get_chunk_end_index(self, unit_word_index):
        unit_word_end_index = unit_word_index + len(self.UNIT_WORD)
        chunk_end_index = unit_word_end_index + self.EXTRA_CHARS_TO_EXTRACT_AROUND_UNIT_WORD
        
        # catch: if required extra chars are not available after unit_word_end_index then while extracting chunk from the content, by default the chunk will end when the content will end. No matter if chunk_end_index is greater than the length of the content.
        return chunk_end_index


    def extract_unit_val_start_index(self, chunk: str):
        unit_word_index_in_chunk = chunk.find(self.UNIT_WORD)
        unit_val_start_index = unit_word_index_in_chunk + len(self.UNIT_WORD)
        return unit_val_start_index


    def extract_unit_val_end_index(self, chunk: str):
        for val in self.UNIT_VAL_ENDING_VALS:
            if val in chunk:
                unit_val_end_index = chunk.find(val)
                return unit_val_end_index
            else:
                pass
        
        unit_val_end_index = len(chunk)
        return unit_val_end_index


    def is_found_invalid_index(self, index):
        return True if (index == -1) else False
    

    def check_for_missing_unit_and_content_vals(self, df: pd.DataFrame):
        """Checks how many rows don't have unit no. or content information. If all rows don't have, then raises exception."""
        num_rows = df.shape[0]
        pages_with_no_unit_info = 0
        pages_with_no_content = 0

        for _, row in df.iterrows():
            unit_no = row['unit_no']
            content = row['content']
            if not unit_no.isnumeric():
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

    def process_book_pdf_and_save_df(self, book_pdf_path, contain_images: bool = False):
        book_pdffile_name = self.extract_book_pdffile_name_from_path(book_pdf_path)
        subject, book_class = self.extract_subject_and_class_from_book_pdffile_name(book_pdffile_name)
        df = self.extract_details_from_book_pdf_path_as_df(book_pdf_path, subject, book_class, contain_images)
        self.save_book_df(df, book_pdffile_name)


    def extract_book_pdffile_name_from_path(self, book_pdf_path):
        book_pdffile_name = os.path.basename(book_pdf_path)
        return book_pdffile_name


    def extract_subject_and_class_from_book_pdffile_name(self, book_name):
        """Book pdffile name is supposed to be like: 'Physics 9', so it separates using space and return the values. The class name should be isnumeric."""
        if not ' ' in book_name:
            raise Exception(f"Book name doesn't contain any space. It is: {book_name}")
        
        if not self.name_ends_with_numeric_char(book_name):
            raise Exception(f"Book name doesn't class i.e. it doesn't end with a numeric character.")
        
        subject, book_class = book_name.split(" ")
        return (subject, book_class)
    
    def name_ends_with_numeric_char(self, name: str):
        end_char = name[-1]
        return True if end_char.isnumeric() else False


    def extract_details_from_book_pdf_path_as_df(self, book_pdf_path, subject: str, book_class: str, contain_images: bool = False):
        if is_valid_pdf_path(book_pdf_path):
            docs = load_single_pdf(book_pdf_path, contain_images)
            df = DOC_PROCESSOR.extract_details_from_docs_as_df(docs, subject, book_class)
            return df


    def save_book_df(self, df, book_name):
        if not self.csv_files_dir_exist():
            os.makedirs(self.csv_dirname, exist_ok=True)

        book_save_path = os.path.join(self.csv_dirname, book_name + ".csv")
        df.to_csv(book_save_path, index=False)
        print(f"Book saved as: {book_save_path}")

        
    def csv_files_dir_exist(self):
        if os.path.exists(self.csv_dirname):
            return True
        else:
            return False



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