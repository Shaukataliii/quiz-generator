import os
from langchain_codebase.codebase import *
import pandas as pd
import ast
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


QUIZ_GENERATOR_PROMPT = """
You will receive the contents of a particular chapter of a student's book. Your job is to create {no_questions} multiple choice question answers with {difficulty_level} difficulty using those contents and provide them in the form of python list where:
1- All are questions are inside that list.
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

    def extract_details_from_docs_as_df(self, docs, book_name):
        pages_with_no_unit_info = 0
        book_data = {
            'book_name': [],
            'page_no': [],
            'unit_no': [],
            'content': []
        }
        for doc in docs:
            page_no, unit_no, content = self.extract_page_no_unit_no_page_content_from_doc(doc)
            book_data['page_no'].append(page_no)
            book_data['unit_no'].append(unit_no)
            book_data['content'].append(content)
            book_data['book_name'].append(book_name)

            if not unit_no.isnumeric():
                pages_with_no_unit_info += 1

        print(f"Pages with no unit info: {pages_with_no_unit_info}")
        return pd.DataFrame(book_data)

    def extract_page_no_unit_no_page_content_from_doc(self, doc):
        page_no = doc.metadata['page']
        page_content = doc.page_content
        unit_no = self.extract_unit_no_from_content(page_content)

        return (page_no, unit_no, page_content)

    def extract_unit_no_from_content(self, content: str):
        print("Extracting unit no...")
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
    


doc_processor = DocumentProcessor()
def extract_details_from_book_pdf_path_as_df(book_pdf_path, book_name: str):
    if is_valid_pdf_path(book_pdf_path):
        docs = load_single_pdf(book_pdf_path)
        df = doc_processor.extract_details_from_docs_as_df(docs, book_name)
        return df
    


import tiktoken
class Tokenizer:
    def get_content_token_count(self, content: str):
        content_tokens = self.tokenize_content(content)
        return len(content_tokens)

    def tokenize_content(self, content: str):
        encoder = tiktoken.get_encoding('gpt2')
        tokens = encoder.encode(content)
        return tokens
    

class QuizGenerator:
    def load_ollama_model(model: str = 'gemma2:2b'):
        return OllamaLLM(model=model)

    def get_response_from_model(llm, prompt):
        return llm.invoke(prompt)

    def get_quiz_generator_prompt(self):
        prompt = PromptTemplate(
            input_variables = ['no_questions', 'difficulty_level', 'book_content'],
            template = QUIZ_GENERATOR_PROMPT
        )
        return prompt


    def load_dataframe(df_filepath: str):
        if QuizGenerator.is_valid_csv_path(df_filepath):
            df = pd.read_csv(df_filepath)
            return df
        
    def is_valid_csv_path(path: str):
        if is_valid_path(path):
            if path[-3:] == 'csv':
                return True
            else:
                raise Exception(f"Not a valid csv file: {path}")
            
    def fetch_relevant_docs_from_vectorstore(vectorstore: Chroma, topic: str, class_no: int, subject: str, unit_no: int, no_docs: int = 3):
        """Uses the provided values in search filters and then performs similarity search and return fetched documents."""
        filters = {
            '$and' : [
                {'class': {'$eq': class_no}},
                {'subject': {'$eq': subject}},
                {'unit_no': {'$eq': unit_no}}
            ]
        }
        similar_docs = vectorstore.similarity_search(topic, k=no_docs, filter=filters)
        return similar_docs
    
    def extracts_contents_from_df_as_list():
        """Requires unit dataframe and returns the pages contents as list."""
        required_df = QuizGenerator.get_provided_unit_df()
        required_content = list(required_df['content'])
        return required_content

    def get_provided_unit_df(df: pd.DataFrame, class_no: int, subject: str, unit: int):
        """Requires class and unit. Loads df ( will use cached using streamlit in future ) and returns the dataframe of that unit."""
        required_df = df[(df['class'] == class_no) & (df['subject'] == subject) & (df['unit_no'] == unit)]
        return required_df

    def evaluate_generated_questions(response: str):
        """Removes keyword "python" and the backticks from response and then evaluates to a python object e.g. list, dict etc. and returns that."""
        response = response.replace("python", '')
        response = response.replace("```", '')
        response = ast.literal_eval(response)
        return response



