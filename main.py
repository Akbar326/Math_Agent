# type: ignore
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from agents.run import RunConfig
import os 
from dotenv import load_dotenv
import math

set_tracing_disabled(disabled=True)
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)

@function_tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@function_tool
async def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@function_tool
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@function_tool
async def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@function_tool
async def modulus(a: int, b: int) -> int:
    """Return the remainder when a is divided by b."""
    return a % b

@function_tool
async def factorial(n: int) -> int:
    """Calculate the factorial of a number."""
    if n < 0:
        raise ValueError("The factorial of a negative number does not exist.")
    if n == 0:
        return 1
    return n * await factorial(n - 1)

@function_tool
async def square_root(a: float) -> float:
    """Return the square root of a number."""
    if a < 0:
        raise ValueError("Cannot take square root of a negative number.")
    return math.sqrt(a)

@function_tool
async def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of two numbers."""
    return math.gcd(a, b)

@function_tool
async def lcm(a: int, b: int) -> int:
    """Return the least common multiple of two numbers."""
    return abs(a * b) // math.gcd(a, b)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant who performs mathematical operations.",
    tools=[add, subtract, multiply, divide, modulus, factorial, square_root, gcd, lcm]
)

# Interactive Loop
print("\n📘 Hello! I'm your Math Agent.")
print("💡 Try operations like: +, -, *, /, %, √ (square root), GCD, LCM")

while True:
    query = input("\nEnter a mathematical question or type 'exit' to end the session: ")
    if query.lower() == 'exit':
        print("Goodbye!")
        break
    try:
        result = Runner.run_sync(agent, query, run_config=config)
        print("Answer:", result.final_output)
    except Exception as e:
        print(f"Error: {e}. Try a different question, like 'What is 6 * 9?'")
