import streamlit as st
from dotenv import load_dotenv
import os

from utils import *
# Load environment variables and initialize Google API
load_dotenv()
initialize_google_api()

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini💁")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete. You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
            else:
                st.warning("Please upload at least one PDF.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["message"])
        else:
            st.chat_message("user").write(msg["message"])

    user_message = st.chat_input("Type your message here")
    if user_message:
        st.session_state.messages.append({"role": "user", "message": user_message})
        # Use the updated user_input function instead of chain.run
        assistant_response = user_input(user_message)
        st.session_state.messages.append({"role": "assistant", "message": assistant_response})
        st.experimental_rerun()

if __name__ == "__main__":
    main()





# import streamlit as st
# from dotenv import load_dotenv
# import os
# #from utils import get_pdf_text, get_text_chunks, get_vector_store, user_input, initialize_google_api
# from utils import get_pdf_text, get_text_chunks, get_vector_store, user_input, initialize_google_api , get_conversational_chain
# # Load envisronment variables
# load_dotenv()

# # Initialize Google API
# initialize_google_api()

# def main():
#     # Set page configuration with a wide layout and a descriptive title
#     st.set_page_config(page_title="Chat PDF", layout="wide")
#     st.header("Chat with PDF using Gemini💁")

#     # Sidebar for PDF upload and processing
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     try:
#                         raw_text = get_pdf_text(pdf_docs)
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("Processing complete. You can now ask questions.")
#                     except Exception as e:
#                         st.error(f"Error processing PDFs: {e}")
#             else:
#                 st.warning("Please upload at least one PDF.")

#     # Initialize chat history in session state if not already present
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display the chat conversation using Streamlit's chat message interface
#     for msg in st.session_state.messages:
#         if msg["role"] == "assistant":
#             st.chat_message("assistant").write(msg["message"])
#         else:
#             st.chat_message("user").write(msg["message"])

#     # Chat input area at the bottom
#     user_message = st.chat_input("Type your message here")
#     if user_message:
#         # Append the user's message to the conversation
#         st.session_state.messages.append({"role": "user", "message": user_message})
#         # Process the user input and obtain the assistant's response
#         assistant_response = user_input(user_message)
#         st.session_state.messages.append({"role": "assistant", "message": assistant_response})
#         # Rerun the app to update the chat window with the new messages
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()









# import streamlit as st
# from dotenv import load_dotenv
# import os
# from utils import get_pdf_text, get_text_chunks, get_vector_store, user_input, initialize_google_api

# # Load environment variables
# load_dotenv()

# # Initialize Google API
# initialize_google_api()

# # Main Streamlit app
# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     # User question input
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     # File upload and process
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     try:
#                         raw_text = get_pdf_text(pdf_docs)
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("Processing complete. You can now ask questions.")
#                     except Exception as e:
#                         st.error(f"Error processing PDFs: {e}")
#             else:
#                 st.warning("Please upload at least one PDF.")

# # Run the app
# if __name__ == "__main__":
#     main()