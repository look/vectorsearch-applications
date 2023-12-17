from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import sys
import json
import os
import datetime

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

data_path = './data/impact_theory_data.json'
## RETRIEVER
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
client = WeaviateClient(api_key=api_key, endpoint=url)
client.display_properties.append('summary')
available_classes = sorted(client.show_classes())
logger.info(available_classes)

## RERANKER
reranker = ReRanker()

## LLM
model_name = 'gpt-3.5-turbo-0613'
llm = GPT_Turbo(model=model_name, api_key=os.environ['OPENAI_API_KEY'])
token_threshold = 3500
encoding = encoding_for_model(model_name)
make_llm_call = True

# I only have one class, so just pick the first one.
class_name = available_classes[0]

data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
    with st.sidebar:
        guest_input = st.selectbox(
            'Select Guest', 
            options=guest_list, 
            index=None, 
            placeholder='Select Guest',
        )
        alpha_input = st.slider(
            'Alpha for hybrid search (controls merging results)', 
            min_value=0.0, 
            max_value=1.0,
            value=0.3,
            step=0.1,
        )
        retreival_limit = st.slider('Max results', min_value=10, max_value=300, value=50, step=10)
        reranker_topk = st.slider('Reranker top K', min_value=1, max_value=5, value=3, step=1)
        temperature_input = st.slider('LLM Temperature', min_value=0.0, max_value=2.0, value=0.1, step=0.1)


    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            # make hybrid call to weaviate
            guest_filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest_input).todict() if guest_input else None
            hybrid_response = client.hybrid_search(
                query, 
                where_filter=guest_filter, 
                class_name=class_name,
                limit=retreival_limit,
            )
            # rerank results
            ranked_response = reranker.rerank(
                results=hybrid_response, 
                query=query, 
                top_k=reranker_topk,
            )
            # validate token count is below threshold
            valid_response = validate_token_threshold(ranked_response, 
                                                    question_answering_prompt_series, 
                                                        query=query,
                                                        tokenizer=encoding,
                                                        token_threshold=token_threshold, 
                                                        verbose=True)
            
            # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                #creates container for LLM response
                chat_container, response_box = [], st.empty()

                # generate prompt
                prompt = generate_prompt_series(query=query, results=valid_response)
                logger.info(prompt)

                try:
                    # execute chat call to LLM
                    if make_llm_call:
                        for resp in llm.get_chat_completion(
                            prompt=prompt,
                            temperature=temperature_input,
                            show_response=True,
                            stream=True,
                        ):
                            try:
                                with response_box:
                                    content = resp.choices[0].delta.content
                                    if content:
                                        chat_container.append(content)
                                        result = "".join(chat_container).strip()
                                        st.write(f'{result}')
                            except Exception as e:
                                logger.error(e)
                                continue
                except Exception as e:
                    logger.error(e)
        
            # Display search results used in context
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url']
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length']
                time_string = datetime.timedelta(seconds=show_length)
                with col1:
                    st.write( 
                        search_result(
                            i=i, 
                            url=episode_url,
                            guest=hit['guest'],
                            title=title,
                            content=hit['content'], 
                            length=time_string,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.write('\n\n')
                with col2:
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()