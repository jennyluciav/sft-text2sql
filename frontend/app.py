"""
Text to SQL generator demo
"""

####################### IMPORTS ####################################
from langchain.prompts import PromptTemplate
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint, LLMContentHandler
from langchain.chains import create_sql_query_chain
from langchain.sql_database import SQLDatabase

import streamlit as st
import ai21
import pandas as pd
import psycopg2
import boto3

import random
import os
from typing import Dict
import json


##################### CONSTANTS ####################################

ENDPOINT_NAME = "j2-grande-instruct-g5-12" # TODO change depending on your endpoint name
EXAMPLE_PROMPTS = [
    "What is Velma's employee id?",
    "How many hours did Peter work in August 2022?",
    "Who worked the most hours in May 2022?",
    "How many Software Engineers does the company have?",
    "Who are the SDEs?",
    "Who are the Employees of the company?",
    "List all Software Engineers who have Peter as their manager",
    "Who are the Software Engineers working on the 'Restaurant Management App' project?"
]
REGION = 'us-east-1'
os.environ['AWS_DEFAULT_REGION'] = REGION # set region to us-east-1 because endpoint is there

TEMPLATE = """Given an input question, create a syntactically correct {dialect} query to run.
Use the following format:

Question: "Question here"
SQLQuery:
"SQL Query to run"

Only use the following tables:

{table_info}.

Some examples of SQL queries that correspond to questions are:

{few_shot_examples}

Question: {input}"""


CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "few_shot_examples", "table_info", "dialect"], template=TEMPLATE
)

FEW_SHOT_EXAMPLES = """

Question: Find what is Peter's email adress.
SQL Query:
SELECT email FROM employees WHERE first_name='Peter';

##

Question: How many Software Engineers does the company have?
SQL Query:
SELECT COUNT(*) from employees
WHERE designation='Software Engineer';

##

Question: How many hours did Velma work in July 2022?
SQL Query:
SELECT SUM(t.entered_hours) AS total_hours_worked
FROM employees e
JOIN timelog t ON e.employee_id = t.employee_id
WHERE e.first_name = 'Velma'
  AND EXTRACT(YEAR FROM t.working_day) = 2022
  AND EXTRACT(MONTH FROM t.working_day) = 7;

##

Question: Who is working on the Music generator project?
SQL Query:
SELECT * FROM employees
WHERE project_id=(
SELECT project_id FROM projects
WHERE project_name = 'Music generator'
);

##

Question: Who works under Max?
SQL Query:
SELECT * FROM employees
WHERE manager_id=(
SELECT employee_id FROM employees
WHERE first_name = 'Max');

##

Question: Who worked the most in April 2022?
SQL Query:
SELECT e.first_name, e.last_name, SUM(t.entered_hours) AS total_hours_worked
FROM employees e
JOIN timelog t ON e.employee_id = t.employee_id
WHERE EXTRACT(YEAR FROM t.working_day) = 2022
  AND EXTRACT(MONTH FROM t.working_day) = 4
GROUP BY e.employee_id, e.first_name, e.last_name
ORDER BY total_hours_worked DESC
LIMIT 1;

##

"""

##################### FUNCTION DEFINITIONS ###############################

@st.cache_resource
def get_schema_img():
    """Loads the schema image from S3 to display on the UI
    """    
    s3_client = boto3.client('s3')
    s3_client.download_file('mihirtestbucketsandbox', 'schema.png', 'schema.png')


def init_connection():
    """Initializes a read-only connection with our RDS PostgreSQL instance.

    Returns:
        psycopg2.connection: connection object to out RDS instance.
    """    
    ssm_client = boto3.client("ssm")
    endpoint=ssm_client.get_parameter(Name="Text2SQLDBEndpoint", WithDecryption=True)["Parameter"]["Value"]
    port=ssm_client.get_parameter(Name="Text2SQLDBPort")["Parameter"]["Value"]
    user=ssm_client.get_parameter(Name="Text2SQLDBUser", WithDecryption=True)["Parameter"]["Value"]
    password=ssm_client.get_parameter(Name="Text2SQLDBPassword", WithDecryption=True)["Parameter"]["Value"]
    conn = psycopg2.connect(host=endpoint, port=port, user=user, password=password)
    conn.set_session(readonly=True)
    return conn

# Perform query.
def execute_query(query):
    """This function executes the given query on a database.

    Args:
        query (str): The SQL query which should be run on a DB.

    Returns:
        pd.DataFrame: The result of that SQL query as a pandas dataframe.
    """
    conn = init_connection()
    try:
        with conn.cursor() as cursor:
            try:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall())
                df.columns = [desc[0] for desc in cursor.description]
                conn.close()
                return df
            except Exception as e:
                st.error('An error occured! - {}'.format(e), icon="🚨")
                conn.rollback()
                # conn.close()
    except psycopg2.InterfaceError as e:
        st.error('{} - Error connecting with the database'.format(e))
        conn.close()



class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"prompt": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["completions"][0]["data"]["text"]

    

def text2sql(query):
    content_handler = ContentHandler()
    parameters = {"maxTokens": 200, "temperature": 0, "numResults": 1}
    llm_ai21 = SagemakerEndpoint(
        endpoint_name=ENDPOINT_NAME,
        region_name=REGION,
        model_kwargs=parameters,
        content_handler=content_handler,
    )
    
    ssm_client = boto3.client("ssm")
    RDS_PORT=ssm_client.get_parameter(Name="Text2SQLDBPort")["Parameter"]["Value"]
    RDS_USERNAME=ssm_client.get_parameter(Name="Text2SQLDBUser", WithDecryption=True)["Parameter"]["Value"]
    RDS_PASSWORD=ssm_client.get_parameter(Name="Text2SQLDBPassword", WithDecryption=True)["Parameter"]["Value"]
    RDS_DB_NAME = "" 
    RDS_ENDPOINT = ssm_client.get_parameter(Name="Text2SQLDBEndpoint", WithDecryption=True)["Parameter"]["Value"]
    RDS_URI = f"postgresql+psycopg2://{RDS_USERNAME}:{RDS_PASSWORD}@{RDS_ENDPOINT}:{RDS_PORT}/{RDS_DB_NAME}"
    db = SQLDatabase.from_uri(RDS_URI,
                           include_tables=["employees", "projects", "timelog"],
                           sample_rows_in_table_info=4)
    prompt = CUSTOM_PROMPT.format(
        input=query,
        table_info=db.table_info,
        dialect="PostgreSQL",
        few_shot_examples=FEW_SHOT_EXAMPLES
    )

    chain = create_sql_query_chain(llm_ai21, db)
    result = chain.invoke({"question": prompt})
    if "##" in result:
        result = result.split("##")[0].split(";")[0] + ";"
    else:
        result = result.split(";")[0] + ";"
    return result

def explain_result(query, result):
    instruction = f"""
    I am building a Natural Language to SQL project. Please formulate an answer to my question in natural language in a human readable format.

    Query: 
    List all the software engineers. 
    Response: 
    [('Peter', 'Kabel', 'Software Engineer'), ('Max', 'Mustermann', 'Software Engineer'), ('Fidel', 'Wind', 'Software Engineer')]
    Explanation:
    The Software Engineers are Peter Kabel, Max Mustermann and Fidel Wind.

    Query: 
    How many hours did Peter work in August 2022?
    Response: 
    [(119,)]
    Explanation:
    Peter worked a total of 119 hours in August 2022.

    Query: 
    List all the projects.
    Response: 
    [(283921, 'Restaurant Management App', 'The Mozzarella Fellas'), (131032, 'Garden Planner', 'Flamingo Gardens'), (933012, 'Music generator', 'ElvisAI'), (311092, 'Weather forecasting system', 'Flamingo Gardens')]
    Explanation:
    Above is a list of all the projects of the company.

    Query:
    {query}
    Response: 
    {result}
    Explanation:
    """
    response = ai21.Completion.execute(sm_endpoint=ENDPOINT_NAME,
                                    prompt=instruction,
                                    maxTokens=80,
                                    temperature=0,
                                    numResults=1)

    return response['completions'][0]['data']['text']

######################## STREAMLIT UI ################################### 

st.set_page_config(layout="wide") # set layout to wide so we can use 2 columns
# set the title
st.title("Natural language to SQL:")

# create a container so we can split in containers
app_container = st.container()
col1, col2 = st.columns([1, 1])
with app_container:
    with col1: # left column
        st.caption("""
           This Generative AI demo lets you query structured data from Amazon Relational Database Service (RDS) with natural language.
            You can now easily consume your data without any knowledge of SQL in a chat like manner. \n
            Give it a try with our pre-configured examples!
           """)
        
        # Load premade example for easier usability
        if st.button(label="Example"):
            st.session_state.prompt = random.choice(EXAMPLE_PROMPTS)

        # first textbox where the user will enter their query in natural language
        st.text_area("Your query in natural language :female-technologist: :male-technologist:", key="prompt")


        # call the function to generate a sql query
        if st.button(label="Generate SQL") and st.session_state.prompt:
            # use a prompt to generate an sql query from the given user input
            prompt = st.session_state.prompt 
            try:
                query = text2sql(prompt)
                st.session_state.results = query
                st.session_state.query = query
            except Exception as e:
                err=f"Unable to call model or interact with database - {e}"
                st.error(err, icon="🚨")
        try:
            if not st.session_state.prompt or not st.session_state.query:
                st.session_state.query = ""
        except AttributeError:
            st.session_state.query = ""


        # Display the query and let the user edit it per needs
        st.write("Generated SQL query :computer:")
        st.code(st.session_state.query, language="sql", line_numbers=True)


        # Let the user download the query
        st.download_button(
            label="Download query",
            data=st.session_state.query.encode('utf-8'),
            file_name='query.sql',
            mime='text/plain',
            )

    with col2: # right column
        # demo db schema
        get_schema_img()
        expander = st.expander("See DemoDB schema")
        expander.write("This is a demo database, where your query can be executed on!")

        expander.image("./schema.png")

        if st.button(label="Execute query") and st.session_state.query:
            df = execute_query(st.session_state.query)
            if df is not None:
                st.subheader("Query results:")
                table = df.style.format(precision=0, thousands='') # set this so streamlit doesn't mistake integers for floats
                st.table(table)
                exp = explain_result(query=st.session_state.prompt, result=df.values.tolist())
                st.subheader("Answer:")
                st.write(exp)
            else:
                st.error("Unable to execute query.", icon="🚨")
