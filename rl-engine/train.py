from .agent import Agent

def get_state():
    return {
        "learningRate": 0.05,
        "explorationRate": 0.1,
        "preferredRegion": "us-east-1"
    }

def apply_update(action):
    pass

def main():
    agent = Agent()
    agent.run(get_state, apply_update, interval_sec=60)

if __name__ == "__main__":
    main()

