import operator
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph

# --- Environment Setup ---
# Set your API keys and email credentials as environment variables.


# this state will be shared across all agents and the orchestrator.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    language: str
    recipient_email: str
    sender_name: str
    research_info: str | None
    draft_email: str | None
    translated_email: str | None
    final_email: str | None
    confirmation_message: str | None


# simple helper function to create each agent.
def create_agent(llm, system_message: str, tools: list = None):
    """Helper function to create a LangChain agent runnable."""
    tools = tools or []
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        return prompt | llm_with_tools
    return prompt | llm


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
search_tool = [TavilySearch(max_results=2)]

# agent 1: researches the topic
research_agent = create_agent(
    llm,
    "You are a research assistant. Your job is to use the search tool to find information on a given topic.",
    tools=search_tool,
)

# agent 2: drafts the email
drafting_agent = create_agent(
    llm,
    "You are an expert email drafter. Your job is to write a compelling first draft of an email about a given topic, based on the research provided.",
)

# agent 3: translates the email
translation_agent = create_agent(
    llm,
    "You are an expert translator. Your job is to translate the given text into the specified language. Default to English if no language is provided.",
)

# agent 4: refines the email
refining_agent = create_agent(
    llm,
    "You are an expert email editor. Your job is to refine a draft into a polished, final version, keeping it in its original language. Your output MUST be in the format: Subject: [Your Subject]\n\n[Your Email Body]",
)


# these nodes will be the steps in the orchestrated workflow.
def research_node(state: AgentState):
    """This node runs the research agent and executes the tool."""
    print("--- 🔬 RESEARCHING TOPIC ---")
    message = HumanMessage(
        content=f"Find information on the following topic: {state['topic']}"
    )
    result = research_agent.invoke({"messages": [message]})

    tool_outputs = []
    if result.tool_calls:
        tool_executor = TavilySearch(max_results=2)
        for tool_call in result.tool_calls:
            output = tool_executor.invoke(tool_call["args"])
            tool_outputs.append(str(output))

    return {"research_info": "\n".join(tool_outputs)}


def drafting_node(state: AgentState):
    """This node runs the drafting agent."""
    print("--- 📝 DRAFTING EMAIL ---")
    content = (
        f"Based on the following research, write an email about '{state['topic']}'.\n"
        f"Use this name '{state['sender_name']}' to sign the email.\n"
        f"This is the researched info:\n\n{state['research_info']}"
    )
    message = HumanMessage(content=content)
    result = drafting_agent.invoke({"messages": [message]})
    return {"draft_email": str(result.content)}


def translation_node(state: AgentState):
    """This node runs the translation agent."""
    print(f"--- 🌐 TRANSLATING TO {state['language'].upper()} ---")
    content = (
        f"You MUST translate the following email draft into the '{state['language']}' language. "
        f"Do not respond in English unless the target language is English. "
        f"Return only the translated text. Here is the draft:\n\n{state['draft_email']}"
    )
    message = HumanMessage(content=content)
    result = translation_agent.invoke({"messages": [message]})
    return {"translated_email": str(result.content)}


def refining_node(state: AgentState):
    """This node runs the refining agent."""
    print("--- ✨ REFINING EMAIL ---")
    content = (
        f"You MUST refine the following email draft, keeping it in the '{state['language']}' language. "
        f"Ensure the final version is polished and professional. "
        f"Here is the draft:\n\n{state['translated_email']}"
    )
    message = HumanMessage(content=content)
    result = refining_agent.invoke({"messages": [message]})
    return {"final_email": str(result.content)}


def sending_node(state: AgentState):
    """This node uses smtplib to send the final email."""
    print("--- 📧 SENDING EMAIL ---")
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    recipient_email = state["recipient_email"]
    final_email_content = state["final_email"]

    if not all([sender_email, sender_password, recipient_email, final_email_content]):
        return {
            "confirmation_message": "Error: Missing required information to send email."
        }

    try:
        subject, body = final_email_content.split("\n\n", 1)
        subject = subject.replace("Subject: ", "").strip()
    except ValueError:
        subject = "No Subject"
        body = final_email_content

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            confirmation = f"Email successfully sent to {recipient_email}!"
            print(confirmation)
            return {"confirmation_message": confirmation}
    except (smtplib.SMTPException, ssl.SSLError) as e:
        error_message = f"Failed to send email: {e}"
        print(error_message)
        return {"confirmation_message": error_message}


workflow = StateGraph(AgentState)

# add the nodes for each agent
workflow.add_node("researcher", research_node)
workflow.add_node("drafter", drafting_node)
workflow.add_node("translator", translation_node)
workflow.add_node("refiner", refining_node)
workflow.add_node("sender", sending_node)

# define the edges that control the flow in a linear sequence
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "drafter")
workflow.add_edge("drafter", "translator")
workflow.add_edge("translator", "refiner")
workflow.add_edge("refiner", "sender")
workflow.add_edge("sender", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Starting the email generation process...")
    topic = input("Please enter the email topic: ")
    language = (
        input(
            "Please enter the target language (e.g., Spanish, French, default is English): "
        )
        or "English"
    )
    recipient_email = input("Please enter the recipient's email address: ")
    sender_name = (
        input("Please enter the name to be used in signing the email: ") or "AI Writer"
    )

    initial_state = {
        "topic": topic,
        "language": language,
        "recipient_email": recipient_email,
        "sender_name": sender_name,
        "messages": [],
    }

    for output in app.stream(initial_state, stream_mode="values"):
        if output.get("confirmation_message"):
            print("\n--- ✅ FINAL CONFIRMATION ---")
            print(output["confirmation_message"])
