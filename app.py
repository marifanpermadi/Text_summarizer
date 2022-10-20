from ast import Num
from turtle import pd
import streamlit as st
import streamlit.components.v1 as stc

from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import pandas as pd
import seaborn as sns
import altair as alt

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import nltk
nltk.download('punkt')

import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
from textblob import TextBlob

import neattext as nt
import neattext.functions as nfx

from wordcloud import WordCloud

from collections import Counter
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

from rouge import Rouge



#rogue eval fx
def eval_summ(summary,reff):
    r = Rouge()
    eval_score = r.get_scores(summary,reff)
    eval_score_df = pd.DataFrame(eval_score[0])
    return eval_score_df


#lex_rank fx
def summy_summ(docx,num=2):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_sum = LexRankSummarizer()
    summary = lex_sum(parser.document,num)
    summ_list = [str(sentence)for sentence in summary]
    result = ' '.join(summ_list)
    return result

def text_analyzer(my_text):
    docx = nlp(my_text)
    allData = [(token.text,token.shape_,token.pos_,token.tag_,
    token.lemma_,token.is_alpha,token.is_stop) for token in docx]
    df = pd.DataFrame(allData,columns=['Token','Shape','PoS','Tag',
    'Lemma','Is_Alpha','Is_Stopword'])
    return df

def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text,entity.label_) for entity in docx.ents]
    return entities

HTML_WRAPPER ="""<div style="overflow-x: auto; border: 1px solid #e6e9ef; 
border-radius: 0.25rem; padding: 1rem">{}</div>"""

def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx,style="ent")
    html = html.replace("\n\n","\n")
    result = HTML_WRAPPER.format(html)
    return result

#most common fx
def get_most_tokens(my_text,num=5):
    word_tokens = Counter(my_text.split())
    most_tokens = dict(word_tokens.most_common(num))
    return most_tokens

#sentiment fx
def get_sentiment(my_text):
    blob = TextBlob(my_text)
    sentiment = blob.sentiment
    return sentiment

#wordcloud fx
def plot_wordcloud(my_text):
    wordc = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(wordc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

		with st.expander("Plot Wordcloud"):
                    plot_wordcloud(raw_text)

#download fx
def download_result(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "nlp_result_{}_.csv".format(timestr)
    st.markdown("### Download CSV File ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.title("Text Summarizer App")
    menu = ["Summerizer","Text Analysis","About"]
    sidemenu = st.sidebar.selectbox("Menu",menu)

    if sidemenu == "Summerizer":
        st.subheader("Summarization")
        raw_text = st.text_area("Enter text here")
        if st.button("Summarize"):

            with st.expander("Original Text"):
                st.write(raw_text)

            c1,c2 = st.columns(2)
            with c1:
                with st.expander("LexRank Summary"):
                    summ = summy_summ(raw_text)
                    doc_len = {"Original":len(raw_text),"Summary":len(summ)}
                    st.write(doc_len)
                    st.write(summ)

                    st.info("Rouge Score")
                    eval_df = eval_summ(summ,raw_text)
                    st.dataframe(eval_df.T)

                    eval_df['metrics'] = eval_df.index
                    c = alt.Chart(eval_df).mark_bar().encode(
                        x='metrics', y='rouge-1')
                    st.altair_chart(c)



            with c2:
                with st.expander("TextRank Summary"):
                    summ = summarize(raw_text)
                    doc_len = {"Original":len(raw_text),"Summary":len(summ)}
                    st.write(doc_len)
                    st.write(summ)

                    eval_df['metrics'] = eval_df.index
                    c = alt.Chart(eval_df).mark_bar().encode(
                        x='metrics', y='rouge-1')
                    st.altair_chart(c)


    if sidemenu == "Text Analysis":
        st.subheader("Analysis")
        raw_text = st.text_area("Enter text here")
        num_of_most_common = st.sidebar.number_input("Most Common Tokens",5,15)
        if st.button("Analyze"):

            with st.expander("Original Text"):
                st.write(raw_text)
            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)
            with st.expander("Entities"):
                # entity_result = get_entities(raw_text)
                entity_result = render_entities(raw_text)
                stc.html(entity_result,height=1000,scrolling=True)

            col1,col2 = st.columns(2)
            with col1:
                with st.expander("Word Stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())
                with st.expander("Top Keywords"):
                    st.info("Top Keywords / Tokens")
                    proc_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_tokens(proc_text,num_of_most_common)
                    st.write(keywords)
                with st.expander("Sentiment"):
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)

            with col2:
                with st.expander("Plot Word Freq"):
                    fig = plt.figure()
                    # sns.countplot(token_result_df['Token'])
                    top_keywords = get_most_tokens(proc_text,num_of_most_common)
                    plt.bar(keywords.keys(),top_keywords.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                with st.expander("Plot Part of Speech"):
                    fig = plt.figure()
                    sns.countplot(token_result_df['PoS'])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                with st.expander("Plot Wordcloud"):
                    plot_wordcloud(raw_text)

            with st.expander("Download Text Analysis Results"):
                download_result(token_result_df)






if __name__ == '__main__':
    main()
