from crewai import Agent, Task, Crew, Process
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

from langchain.tools import tool
import pandas as pd

from langchain.llms import Ollama
from langchain_openai import ChatOpenAI
#ollama_llm = Ollama(model="phi")

load_dotenv(override=True)
load_dotenv()


openai_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_key



@tool
def parse_transactions(file_path) -> pd.DataFrame:
    """
    Parses a CSV file containing personal credit card transactions and returns a structured DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the structured credit card transactions.
    """
    transactions_df = pd.read_csv('/workspaces/codespaces-jupyter/data/Credit_Karma_Transactions_Sample.csv')

    return transactions_df.to_dict(orient='records') 

@tool
def analyze_and_recommend(transactions: list) -> str:
    """
    Analyzes personal credit card transactions to provide recommendations on savings and expense reductions.
    
    Args:
        transactions_df (pd.DataFrame): A DataFrame containing structured credit card transactions.
        
    Returns:
        str: Recommendations on savings and areas to cut down expenses.
    """
    # Example analysis (to be customized based on actual transaction categories and logic)
    transactions_df = pd.DataFrame(transactions)
    monthly_spending = transactions_df.groupby('Category')['Amount'].sum()
    recommendations = []

    # Identify high spending categories
    high_spending_categories = monthly_spending[monthly_spending > monthly_spending.median()].index.tolist()
    for category in high_spending_categories:
        recommendations.append(f"Consider reducing expenses in '{category}' as it's above your median spending.")
    
    # Suggest savings based on analysis
    recommendations.append("Review recurring subscriptions and memberships for services not frequently used.")
    
    return '\n'.join(recommendations)




# Define the Personal Financial Planner agent
personal_financial_planner = Agent(
    role='Personal finance adviser',
    goal='Analyze expenses and provide insights for savings enhancement',
    backstory='An expert in financial data analysis, specialized in identifying cost-cutting opportunities to enhance savings.And provide recommendations for cost reduction by looking at spending categories.',
    tools=[parse_transactions],
    verbose=True,
    allow_delegation=True,
    #llm=ollama_llm 
    llm=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.7) # Assuming these tools are registered and available
    
)

task1 = Task(description='Process CSV file and analyze spending.', agent=personal_financial_planner)
task2 = Task(description='Based on the input data provided ,Analyze category-wise spending based on the input data.and provide insisght on how much spent on each category,give the percentage of spending ', agent=personal_financial_planner)
task3 = Task(description='based on the data that was categorized ,Provide recommendations for cost reduction by looking at spending categories and compare with mean', agent=personal_financial_planner)



# Configure the crew
crew = Crew(
    agents=[personal_financial_planner],
    tasks=[task1, task2, task3],
    verbose=2,
    #process=Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)

# Note: This example assumes the presence of a 'Category' and 'Amount' column in the transactions DataFrame.
# You may need to adjust the analysis logic based on the actual structure of your transactions data.
