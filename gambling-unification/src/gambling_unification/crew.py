# File: src/gambling_unification/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from litellm import completion
import os
import logging

# Configure LiteLLM for Ollama
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["LITELLM_MODEL"] = "ollama/llama3"

class GamblingTools:
    @staticmethod
    def scrape_polymarket():
        """Mock scraper implementation"""
        return [{"product": "Election 2024", "price": 0.45}]
    
    @staticmethod
    def analyze_products(data):
        """LLM-powered product matching"""
        try:
            response = completion(
                model=os.getenv("LITELLM_MODEL"),
                messages=[{
                    "role": "user",
                    "content": f"Analyze these products:\n{data}"
                }],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return None

@CrewBase
class GamblingUnificationCrew():
    """Gambling market unification crew"""

    @agent
    def data_collector(self) -> Agent:
        return Agent(
            config=self.agents_config['data_collector'],
            tools=[GamblingTools.scrape_polymarket],
            verbose=True
        )

    @agent 
    def product_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['product_analyst'],
            tools=[GamblingTools.analyze_products],
            verbose=True
        )

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_engineer'],
            verbose=True
        )

    @task
    def data_collection_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_collection_task'],
            agent=self.data_collector
        )

    @task
    def product_matching_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_matching_task'],
            agent=self.product_analyst,
            context=[self.data_collection_task]
        )

    @task
    def report_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_generation_task'],
            agent=self.data_engineer,
            context=[self.product_matching_task],
            output_file='results/unified_products.csv'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=2
        )