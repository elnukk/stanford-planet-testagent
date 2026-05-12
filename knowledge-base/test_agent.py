from dspy_agent import build_agent

def main():
    agent = build_agent()

    response = agent.forward(
        user_request="How do I authenticate with Planet APIs?"
    )

    print("\n===== AGENT RESPONSE =====\n")
    print(response)

if __name__ == "__main__":
    main()