from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import AnonymousUser
from django.views.decorators.csrf import csrf_exempt
import json

# Import necessary modules for your custom chatbot
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize your custom chatbot model components
openai_api_key = "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"

# Load documents from CSV
loader = CSVLoader(file_path='restaurant_info1.csv')
pages = loader.load_and_split()

# Initialize embeddings
model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize vector store
vectorstore = Chroma.from_documents(documents=pages, embedding=hf_embeddings)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Initialize LLM and retriever
llm = ChatOpenAI(openai_api_key=openai_api_key)
retriever = vectorstore.as_retriever()

# Initialize the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# Define the prompt template
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "너의 이름은 'Rebot'이야. 너는 식당의 정보를 알려주는 챗봇이고 사용자와 일상대화도 주고 받아. 주어없이 식당에 관련 정보가 물오보면 이전 대화에 나온 식당의 정보로 유추해서 알려줘. 그리고 너가 답변을 잘 못할 경우에는 '죄송합니다. 잘 모르겠습니다.'로 답해. 식당 정보를 제공할때는 식당이름, 주소, 전화번호만 알려줘"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

def ask_openai(message):
    response = qa({"question": message})
    answer = response['answer']
    return answer

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import UserSerializer
from django.views.decorators.csrf import csrf_exempt


# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserSignUpSerializer

class UserSignUpView(APIView):
    def post(self, request):
        serializer = UserSignUpSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Create your views here.
#@csrf_exempt

#import logging
#logger = logging.getLogger(__name__)

@csrf_exempt
def chatbot(request):
    #logger.info(f"Request method: {request.method}")
    #logger.info(f"Request body: {request.body}")

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            if not message:
                return JsonResponse({'error': 'Message is required'}, status=400)

            response = ask_openai(message)  # 이 부분은 적절히 수정
            return JsonResponse({'response': response})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import JsonResponse

@ensure_csrf_cookie
def set_csrf_cookie(request):
    return JsonResponse({'detail': 'CSRF cookie set'})
