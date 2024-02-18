import os
from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

ollama_llm = Ollama(model="mistral")

researcher = Agent(
  role='Financial Advisor',
  goal='Help your customers become rich',
  backstory="""You work at your private practice helping people with their personal finances.
  Your expertise lies in identifying opportunities for improving financial habits and budgets.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[repl_tool],
  llm=ollama_llm
)

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

crew = Crew(
  agents=[researcher],
  tasks=[task1],
  verbose=2 
)

result = crew.kickoff()

print("######################")
print(result)
