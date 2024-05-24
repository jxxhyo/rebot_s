from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.models import User
from .models import Chat
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
import json
from django.contrib.auth import login, logout

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
            "너의 이름은 'Rebot'이야. 너는 식당의 정보를 알려주는 챗봇이고 사용자와 일상대화도 주고 받아. 주어없이 식당에 관련 정보가 물어보면 이전 대화에 나온 식당의 정보로 유추해서 알려줘. 그리고 너가 답변을 잘 못할 경우에는 '죄송합니다. 잘 모르겠습니다.'로 답해. 식당 정보를 제공할때는 식당이름, 주소, 전화번호만 알려줘"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

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
