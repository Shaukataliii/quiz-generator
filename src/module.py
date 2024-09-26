import streamlit as st
from langchain_codebase.codebase import *
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract
import re, ast


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
BOOKS_VDB_PATH = "VDB-ALL-BOOKS"

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
    def load_ollama_llm(self, model: str = 'gemma2:2b'):
        return OllamaLLM(model=model)
    
    def load_pandas_df(self, df_path: str):
        if self.is_valid_csv_path(df_path):
            return pd.read_csv(df_path)
        
    def is_valid_csv_path(self, path: str):
        if is_valid_path(path):
            if path[-3:] == 'csv':
                return True
            else:
                raise Exception(f"Not a valid csv file: {path}")
        

    def get_response_from_model(llm, prompt):
        return llm.invoke(prompt)

    def get_quiz_generator_prompt():
        prompt = PromptTemplate(
            input_variables = ['no_questions', 'book_content'],
            template = QUIZ_GENERATOR_PROMPT
        )
        return prompt



# added without cleaning
    


class InputsHandler:
    """
    Handles the inputs using provided dataframe path and provides details i.e. available classes, available subjects and available units depending on selections.

    Provided dataframe should have the following structure:
    0: class_details,
    1: subject_details, 
    2: unit_details
    """
    def __init__(self, details_df_path: str):
        """
        Initializes dataframe. Raises exceptions if path is invalid or is not a csv file.
        """
        details_df = UTILS.load_pandas_df(details_df_path)
        
        self.details_df = details_df
        self.class_key = details_df.columns[0]
        self.subjects_key = details_df.columns[1]
        self.units_key = details_df.columns[2]

    
    def get_supported_classes_list(self) -> list:
        """
        Returns list of classes available in the dataframe.
        """
        supported_classes = self.details_df[self.class_key]
        return supported_classes.unique().tolist()


    def get_selected_class_subjects(self, selected_class: int) -> list:
        """
        From the dataframe, Returns a list of available subjects for the provided class.
        """
        subjects = self._get_class_df(selected_class)[self.subjects_key]
        return subjects.unique().tolist()
    
    
    def get_selected_subject_units(self, selected_class: int, selected_subject: str) -> list:
        """
        From the dataframe, Returns a list of available units for the provided class and subject.
        """
        class_df = self._get_class_df(selected_class)
        subject_df = class_df[class_df[self.subjects_key]==selected_subject]
        units = subject_df[self.units_key]
        return units.unique().tolist()
    
    
    def _get_class_df(self, selected_class: str) -> pd.DataFrame:
        """
        Internal method to filter dataframe for the provided class.
        Raises ValueError if class not found.
        """
        class_df = self.details_df[self.details_df[self.class_key]==selected_class]

        if class_df.empty:
            raise ValueError(f"Class: {selected_class} is not present in dataframe.")
        
        return class_df
    
    

class QuizProcessor:
    """
    Uses the generated quiz and processes it. Provides formatted strings to display the questions and evalute student performance.
    
    :param quiz:
        A list containing question dictionaries having keys:
        - question: the question (str)
        - options: list of options (list)
        - answer: index of correct option (int)
    """
    def __init__(self, quiz: list):
        """Initializes the provided quiz. Raises TypeError is quiz is not a list."""

        if not isinstance(quiz, list):
            raise TypeError("quiz must be a list.")
        
        self.quiz = quiz


    def get_formatted_quiz(self):
        """
        Formats the quiz into a list of questions to use in streamlit. Each question is formated into a list containing:
            - question text
            - dictionary to use for radio button containing keys:
                - label
                - options
                - key : The question key to use. This is to facilitate the evaluation process.

        :return: Quiz containing formatted questions (list).
        """
        quiz = []
        for index, mcq in enumerate(self.quiz):
            q_no = index + 1
            q_key = self._get_question_key(index)

            question = f"Q{q_no}: {mcq['question']}"
            radio = {
                'label': "Select your answer:", 
                'options': mcq['choices'], 
                'key': q_key
                }
            formatted_q = [question, radio]
            quiz.append(formatted_q)

        return quiz
    
    def evaluate_student_performance(self, session) -> tuple:
        """
        Evaluates student answers based on correct answers in quiz.

        :param session: The streamlit session state after user has answers the quiz.
        :return: A tuple containing:
            - Number of correct answers (int)
            - Details of the wrong answers in format:- "Q question_no: correct_choice, ...
        """
        correct_count = 0
        wrong_details = []
        for index, mcq in enumerate(self.quiz):
            q_no = index + 1
            q_key = self._get_question_key(index)
            correct_option_index = mcq['correct_choice']

            answer_text = session[q_key]
            answer_index_in_mcq = mcq['choices'].index(answer_text)

            if answer_index_in_mcq == correct_option_index:
                correct_count += 1
            else:
                detail = f"Q {q_no}: {correct_option_index + 1}"
                wrong_details.append(detail)

        return (correct_count, ", ".join(wrong_details))
        
        
    def all_questions_answered(self, session):
        """
        Checks if all the questions in the quiz are answered.
        For each question, checks if the answer is None which indicates unanswered question.

        :param session: The streamlit session state containing user answers.
        :return: True if all questions are answered else False.
        """
        for index in range(len(self.quiz)):
            q_key = self._get_question_key(index)
            answer_text = session[q_key]

            if answer_text is None:
                return False
        return True
    
    def _get_question_key(self, q_index: int) -> str:
        """
        Generates a key to use in the streamlit session state.

        :param q_index: Index of question in quiz.
        """
        return f"q_{q_index}"
        


class QuizGenerator:
    """
    Generates quizzes based on provided class, subject, unit, and other parameters. 
    It interacts with the Vectorstore and LLM, retrieves relevant documents and generates quiz.
    """
    def __init__(self, selected_class: int, subject: str, unit: int, num_questions: int, difficulty_level: str, topic: str, num_docs: int = 3):
        """
        Initializes QuizGenerator with the necessary parameters.

        :param selected_class: The class number (e.g., 10 for 10th grade).
        :param subject: The subject for which to generate the quiz.
        :param unit: The unit number within the subject.
        :param num_questions: The number of questions to generate in the quiz.
        :param difficulty_level: The difficulty level of the quiz (e.g., easy, medium, hard).
        :param topic: The topic or keyword for quiz generation.
        :param num_docs: The number of relevant documents to retrieve from the vector store (default is 3).
        """
        self.selected_class = selected_class
        self.subject = subject
        self.unit = unit
        self.num_questions = num_questions
        self.difficulty_level = difficulty_level
        self.topic = topic
        self.num_docs = num_docs

        # self.llm = UTILS.load_ollama_llm()
        self.llm = get_google_llm()
        self.vectorstore = load_vectorstore(BOOKS_VDB_PATH)

    def generate_quiz(self) -> list:
        """
        Generates a quiz by fetching relevant documents, constructing a prompt, and querying the LLM.

        :return: The generated quiz as a list of questions.
        :raises ValueError: If no relevant documents are found or the LLM response is invalid.
        """
        try:
            docs = self._get_relevant_docs_as_string()
            if not docs:
                raise ValueError(f"No documents found for the given criteria.")

            prompt = self._get_formatted_prompt(docs)
            quiz = self._generate_quiz_from_prompt(prompt)
            return quiz
        
        except Exception as e:
            raise ValueError(f"Error generating quiz. {str(e)}")
        

    def _get_relevant_docs_as_string(self):
        """
        Fetches relevant documents from the vector store and combines them into a single string.

        :return: Combined text of relevant documents.
        :raises ValueError: If no documents are retrieved.
        """
        docs = self._fetch_relevant_docs_from_vectorstore()
        if not docs:
            raise ValueError(f"No documents found for the given criteria.")
        
        docs = [doc.page_content for doc in docs]
        docs = " ".join(docs)
        return docs

    def _fetch_relevant_docs_from_vectorstore(self):
        """
        Performs a similarity search on the vector store using the provided parameters.

        :return: A list of documents retrieved from the vector store.
        :raises ValueError: If any search parameters are missing or invalid.
        """
        try:
            filters = {
                '$and' : [
                    {'class': {'$eq': self.selected_class}},
                    {'subject': {'$eq': self.subject}},
                    {'unit_no': {'$eq': self.unit}}
                ]
            }
            similar_docs = self.vectorstore.similarity_search(self.topic, k=self.num_docs, filter=filters)
            return similar_docs
        
        except Exception as e:
            raise ValueError(f"Error fetching documents from vector store: {str(e)}")


    def _get_formatted_prompt(self, docs: str):
        """
        Formats the prompt for the LLM by embedding the quiz parameters and document content.

        :param docs: A string containing the combined content of relevant documents.
        :return: The formatted prompt string.
        :raises ValueError: If the prompt template is invalid.
        """
        try:
            prompt = self._get_quiz_generator_prompt()
            prompt = prompt.format(no_questions=self.num_questions, difficulty_level=self.difficulty_level, book_content=docs)
            return prompt
        
        except Exception as e:
            raise ValueError(f"Invalid prompt template: {str(e)}")
        
    def _get_quiz_generator_prompt(self):
        """
        Returns the quiz generation prompt template.

        :return: A PromptTemplate object with the template for quiz generation.
        """
        prompt = PromptTemplate(
            input_variables = ['no_questions', 'difficulty_level', 'book_content'],
            template = QUIZ_GENERATOR_PROMPT
        )
        return prompt


    def _generate_quiz_from_prompt(self, prompt):
        """
        Sends the formatted prompt to the LLM and evaluates the response.

        :param prompt: The formatted prompt string.
        :return: A list of quiz questions.
        :raises ValueError: If the LLM response is malformed or contains invalid data.
        """
        try:
            response = self._get_response_from_llm(prompt)
            response = self._evaluate_generated_quiz(response)
            return response
        
        except Exception as e:
            raise ValueError(f"Error generating quiz from LLM response: {str(e)}")

    def _get_response_from_llm(self, prompt):
        """
        Sends a request to the LLM with the given prompt and returns its response.

        :param prompt: The prompt string to send to the LLM.
        :return: The raw response from the LLM.
        :raises RuntimeError: If the LLM invocation fails.
        """
        try:
            return self.llm.invoke(prompt)
        
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {str(e)}")

    def _evaluate_generated_quiz(self, response: str):
        """
        Processes the LLM response, removing unwanted text and converting it to a Python object.

        :param response: The raw response from the LLM.
        :return: The evaluated response, typically a list of quiz questions.
        :raises ValueError: If the response cannot be parsed as a valid Python object.
        """
        try:
            response = self._remove_unwanted_chars_from_response(response)
            response = ast.literal_eval(response)
            return response
        
        except Exception as e:
            raise ValueError(f"Failed to evaluate generated quiz: {str(e)}")
    
    def _remove_unwanted_chars_from_response(self, response: str):
        """
        Removes 'python' and '```' form response and returns the response.
        """
        response = response.replace("python", '').replace("```", '')
        return response


  



DOC_PROCESSOR = DocumentProcessor()
OCR_USER = OCRUser()
BOOK_PATH_PROCESSOR = BookPathProcessor()
UTILS = Utils()