from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

# template = PromptTemplate(
#     template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
#     input_variables=['place'],
#     partial_variables={'format_instruction':parser.get_format_instructions()}
# )

template = PromptTemplate(
    template=(
        "You are a helpful assistant. Generate ONLY a JSON object with the following fields: name, age, city.\n"
        "The person should be a fictional {place} person.\n"
        "Follow this format exactly:\n{format_instruction}\n"
        "Do not include any extra text, comments, or schema descriptions — only real example values."
    ),
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)