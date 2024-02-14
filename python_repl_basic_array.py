import os
from crewai import Agent, Task, Crew, Process

# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# You can choose to use a local model through Ollama for example. See ./docs/how-to/llm-connections.md for more information.
from langchain.llms import Ollama
from langchain.agents import Tool
# can we get it to choose from all the tools in langchain?
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# Estimate the number of trees in a forest, and then take the square root of that.

# python_repl.run("Estimate the number of trees in a forest, and then take the square root of that.")

ollama_llm = Ollama(model="mistral")

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun
# search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
researcher = Agent(
  role='Financial Advisor',
  goal='Help your customers become rich',
  backstory="""You work at your private practice helping people with their personal finances.
  Your expertise lies in identifying opportunities for improving financial habits and budgets.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[repl_tool],
  # You can pass an optional llm attribute specifying what mode you wanna use.
  # It can be a local model through Ollama / LM Studio or a remote
  # model like OpenAI, Mistral, Antrophic or others (https://python.langchain.com/docs/integrations/llms/)
  #
  # Examples:
  llm=ollama_llm # was defined above in the file
  # llm=OpenAI(model_name="gpt-3.5", temperature=0.7)
  # For the OpenAI model you would need to import
  # from langchain_openai import OpenAI
)
# writer = Agent(
#   role='Tech Content Strategist',
#   goal='Craft compelling content on tech advancements',
#   backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
#   You transform complex concepts into compelling narratives.""",
#   verbose=True,
#   allow_delegation=True,
#   llm=ollama_llm
# )

# Create tasks for your agents
# task1 = Task(
#   description="""Conduct a comprehensive analysis of the spreadsheet.
#   Identify key trends, breakthrough technologies, and potential industry impacts.
#   Your final answer MUST be a full analysis report""",
#   agent=researcher
# )



# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the following numbers. [0.15462267, 0.12245859, 0.85776414, 0.08278679, 0.19400978,
 0.76829536, 0.33164133, 0.73665353, 0.82198991, 0.02057661,
 0.58422872, 0.37010361, 0.52565938, 0.47213661, 0.3935184 ,
 0.30059291, 0.71102641, 0.64490755, 0.03795531, 0.71658756,
 0.00312536, 0.84548975, 0.91045778, 0.20044064, 0.15033803,
 0.30830523, 0.02596581, 0.34319401, 0.89883875, 0.62035665]
 Please write a Python script to analyze these numbers. The script should explicitly print any insights about the input. Ensure that your script uses the `print` function to display the results.
  Identify and output key trends and patterns.
  Your final answer MUST include all findings.""",
  agent=researcher
)

# task2 = Task(
#   description="""Using the insights provided, develop an engaging blog
#   post that highlights the most significant AI advancements.
#   Your post should be informative yet accessible, catering to a tech-savvy audience.
#   Make it sound cool, avoid complex words so it doesn't sound like AI.
#   Your final answer MUST be the full blog post of at least 4 paragraphs.""",
#   agent=writer
# )

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher],
  tasks=[task1],
  verbose=2 # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)