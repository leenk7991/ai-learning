import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph

# --- Environment Setup ---
# make sure you have the following keys in your env
# "GOOGLE_API_KEY" and "TAVILY_API_KEY"


class AgentState(TypedDict):
    # The 'messages' field will hold the conversation history.
    # 'operator.add' specifies that new messages should be appended to the list.
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Tools are the actions our agent can take. Here, we'll give it the ability
# to search the web.
tools = [TavilySearch(max_results=1)]
tool_executor = TavilySearch(max_results=1)

# We'll use the gemini-2.5-flash model as the "brain" of our agent.
# We bind the tools to the model so it knows what actions it can call.
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.7
)  # increased temperature for more creative role-playing
model_with_tools = model.bind_tools(tools)


# Node 1: The main agent logic that decides what to do next.
def agent(state):
    """
    Invokes the language model to decide the next action.
    If the model calls a tool, this function returns a tool call message.
    If the model responds directly, it returns the final answer.
    """
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    # The response from the model is appended to the message history.
    return {"messages": [response]}


# Node 2: A function to execute the tools called by the agent.
def action(state):
    """
    Executes the tool calls identified by the 'agent' node.
    It takes the last message (which should be a tool call), runs the tool,
    and returns the tool's output as a new message.
    """
    last_message = state["messages"][-1]

    # Defensive check for tool calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_output = tool_executor.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}


# Edges determine the path through the graph. A conditional edge uses a
# function to decide which node to go to next based on the current state.
def should_continue(state):
    """
    This function decides the next step after the 'agent' node has run.
    - If the last message is a tool call, it routes to the 'action' node.
    - If there are no tool calls, it means the agent has a final answer,
      and the graph should end.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"  # Go to the tool execution node
    return "end"  # End the graph execution


if __name__ == "__main__":
    # Initialize the graph and define the state object.
    workflow = StateGraph(AgentState)

    # Add the nodes to the graph.
    workflow.add_node("agent", agent)
    workflow.add_node("action", action)

    # Set the entry point of the graph.
    workflow.set_entry_point("agent")

    # Add the conditional edge. After the 'agent' node, the 'should_continue'
    # function will be called to determine the next step.
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "end": END,
        },
    )

    # Add a normal edge. After the 'action' node runs, it always goes back to the 'agent'
    # node to process the tool's output.
    workflow.add_edge("action", "agent")

    # Compile the graph into a runnable object.
    app = workflow.compile()

    # Now we can interact with our compiled agent in a loop.
    print("AI: Hello, what character would you like me to role-play today?")
    persona_prompt = input("You: ")

    # The first message in our history is the System Message that defines the persona.
    conversation_history = [SystemMessage(content=persona_prompt)]

    # Get the character's name
    name_request_inputs = {
        "messages": conversation_history
        + [
            HumanMessage(
                content="Based on the character I've asked you to play, what is your name? Respond with only the name."
            )
        ]
    }
    character_name = "AI"
    for output in app.stream(name_request_inputs):
        for key, value in output.items():
            if key == "agent" and not value["messages"][-1].tool_calls:
                character_name = value["messages"][-1].content.strip()

    # Get the character's introduction
    intro_request_inputs = {
        "messages": conversation_history
        + [HumanMessage(content="Please introduce yourself in character.")]
    }
    ai_response_message = None
    for output in app.stream(intro_request_inputs):
        for key, value in output.items():
            if key == "agent" and not value["messages"][-1].tool_calls:
                latest_message = value["messages"][-1]
                print(f"\n{character_name}: {latest_message.content}")
                ai_response_message = latest_message

    # Add the AI's introduction to the history
    if ai_response_message:
        conversation_history.append(ai_response_message)

    print("\n---")  # Separator for clarity

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "", "goodbye"]:
            print("Exiting agent.")
            break

        # The 'inputs' dictionary will  contain the entire conversation history
        # plus the new user message for this turn.
        current_human_message = HumanMessage(content=user_input)
        inputs = {"messages": conversation_history + [current_human_message]}

        ai_response_message = None

        # The 'stream' method allows us to see the agent's thought process step-by-step.
        for output in app.stream(inputs):
            for key, value in output.items():
                # We only want to print the final response, not the intermediate steps
                if key == "agent" and not value["messages"][-1].tool_calls:
                    latest_message = value["messages"][-1]
                    print(f"\n\n{character_name}: {latest_message.content}\n\n")
                    ai_response_message = latest_message

        # After the stream is complete, update the conversation history
        conversation_history.append(current_human_message)
        if ai_response_message:
            conversation_history.append(ai_response_message)
