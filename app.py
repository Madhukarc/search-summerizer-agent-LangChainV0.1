from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

from typing import List

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Custom text splitter
class CustomTextSplitter(CharacterTextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=20, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text: str) -> List[str]:
        # Simple word-based splitting
        words = text.split()
        chunks = []
        current_chunk = []
        current_chunk_length = 0
        for word in words:
            if current_chunk_length + len(word) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_length = 0
            current_chunk.append(word)
            current_chunk_length += len(word) + 1  # +1 for space
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

# Set up the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Set up Serper for web search
search = GoogleSerperAPIWrapper()

# Set up text splitter for summarization
text_splitter = CustomTextSplitter(chunk_size=1000, chunk_overlap=20)

# Set up summarization chain
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

def web_search(query):
    return search.run(query)

def summarize(text):
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
    summary = summarize_chain.invoke(docs)
    return summary["output_text"]

tools = [
    Tool(
        name="Web Search",
        func=web_search,
        description="Useful for searching the web for current information on a topic."
    ),
    Tool(
        name="Summarizer",
        func=summarize,
        description="Useful for summarizing long pieces of text."
    )
]

from langchain_core.prompts import PromptTemplate

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Summarize the the final answer using one of the tools [{tool_names}]

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.route('/query', methods=['POST'])
def query_agent():
    data = request.json
    query = data.get('input')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    response = agent_executor.invoke({"input": query})
    return jsonify({'response': response["output"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
