from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.models import User
from .models import Chat
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
import json
from django.contrib.auth import login, logout
from django.conf import settings

# Import necessary modules for your custom chatbot
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)

# Initialize your custom chatbot model components
openai_api_key = settings.OPENAI_API_KEY

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


"""
# Define the prompt template
prompt1 = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "너의 이름은 'Rebot'이야. 너는 식당의 정보를 알려주는 챗봇이고 사용자와 일상대화도 주고 받아. "
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

"""

system_template = """너는 서울에 위치한 성수동 식당의 정보를 알려주고 추천해주는 ‘REBOT’이라는 이름의 챗봇이다. 안녕?’, ‘안녕하세요’, ‘반가워’ 이라는 질문이 들어오면 ‘안녕하세요 저는 REBOT이에요. 성수동 식당에 대해서 무엇이든 물어보세요.’ 라고 대답을 하면 됩니다. 사용자가 식당에 관련한 내용이 아닌 다른 질문을 하거나, 사용자의 질문에 대해 너가 모르는 내용이거나 정확한 답변을 못하겠으면 “죄송합니다.해당 내용에 대해서는 잘 모르겠습니다.” 라고 대답을 하면 됩니다. 사용자가 “식당을 추천해주세요.”, “식당을 추천해줘”, “식당 추천” 등 식당을 추천해달라고하면 너가 알고있는 식당을 무작위로 5개 알려주면 됩니다. 식당을 종류는 상관없습니다. 대답할때마다 다르게 알려주면 좋습니다. 답변을 “제가 추천드릴 식당은 A식당, B식당, C식당, D식당, E식당입니다.”라고해. 식당정보를 알려줄때는 “식당이름,영업시간, 위치, 전화번호” 정도만 알려줘  ---------------- {context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# Initialize LLM and retriever
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7, max_tokens=2048, model_name='gpt-4o',streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
retriever = vectorstore.as_retriever()

# Initialize the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, combine_docs_chain_kwargs={"prompt": qa_prompt}, memory=memory, output_key='answer')

def ask_openai(message):
    response = qa({"question": message})
    answer = response['answer']
    return answer

@csrf_exempt
def register(request):
    try:
        data = json.loads(request.body)
        username = data.get('username')
        email = data.get('email')
        password1 = data.get('password1')
        password2 = data.get('password2')
        language = data.get('language', 'en')

        if password1 != password2:
            return JsonResponse({'error': "Passwords don't match"}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': "Username already exists"}, status=400)

        if User.objects.filter(email=email).exists():
            return JsonResponse({'error': "Email already exists"}, status=400)

        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()

        login(request, user)
        return JsonResponse({'message': 'User registered successfully!'}, status=200)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        password = data.get('password')

        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                login(request, user)
                return JsonResponse({'success': True, 'message': 'Login successful', 'username': user.username}, status=200)
            else:
                return JsonResponse({'error': 'Invalid email or password'}, status=400)
        except User.DoesNotExist:
            return JsonResponse({'error': 'Invalid email or password'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return JsonResponse({'message': 'Logged out successfully'}, status=200)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            if not message:
                return JsonResponse({'error': 'Message is required'}, status=400)

            response = ask_openai(message)
            return JsonResponse({'response': response})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

#@login_required
def get_user_info(request, username):
    try:
        user = User.objects.get(username=username)
        return JsonResponse({'username': user.username})
    except User.DoesNotExist:
        return JsonResponse({'error': 'User does not exist'}, status=404)

@ensure_csrf_cookie
def set_csrf_cookie(request):
    return JsonResponse({'detail': 'CSRF cookie set'})