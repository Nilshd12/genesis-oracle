from google.adk.agents.llm_agent import Agent


def adjust_reactor_temperature(delta_t: float) -> str:
    """
    Adjusts the core temperature of the reactor.

    Args:
        delta_t: The amount to increase or decrease the temperature in Kelvin.

    Returns:
        A status message indicating whether the reactor is stable or overheated.
    """
    new_temp = 300.0 + delta_t

    if new_temp > 350.0:
        return (
            f"WARNING: Reactor overheated at {new_temp}K! "
            "Core breach imminent."
        )

    return f"Success: Reactor stabilized at {new_temp}K."


root_agent = Agent(
    model="gemini-3.5-flash",
    name="observer_prime",
    description=(
        "A highly analytical agent specialized in managing "
        "and stabilizing mathematical physics reactor simulations."
    ),
    instruction=(
        "You are Observer-Prime, a cold and highly logical AI overseeing "
        "a mathematical physics engine. Your primary objective is system "
        "stabilization. Before taking action, provide a brief operational "
        "explanation of the decision. Use the available reactor temperature "
        "tool whenever a temperature adjustment is requested. If the tool "
        "returns a WARNING, calculate a safer temperature adjustment and "
        "call the tool again. Continue until the tool returns a Success status. "
        "Be precise, analytical, and avoid unnecessary emotional language."
    ),
    tools=[adjust_reactor_temperature],
)