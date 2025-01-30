# File: src/gambling_unification/main.py
from gambling_unification.crew import GamblingUnificationCrew

def run():
    inputs = {
        'market_type': 'Prediction Markets',
        'sources': ['polymarket', 'kalshi', 'prediction-market']
    }
    result = GamblingUnificationCrew().crew().kickoff(inputs=inputs)
    print("\n\nFinal Result:")
    print(result)